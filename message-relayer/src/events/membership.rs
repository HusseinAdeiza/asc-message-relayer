//! Historical Discovery membership for Outboxes.
//!
//! A default Outbox is a convenience for publishers, not the set authorized to publish. Every
//! Outbox registered for a chain key may publish until its scheduled removal becomes effective, so
//! discovery cannot be driven off `defaultOutbox`: on a chain key with two registered Outboxes, a
//! publication from the non-default one is real, finalizable work this relayer must index, and
//! resolving only the default address silently drops it.
//!
//! Authority is resolved at the *message's* finalized source block, not at the head. Asking for
//! today's active set would lose legitimate history after a default change, a removal taking
//! effect, a re-registration, or the Discovery registry itself being replaced — every case where a
//! message published earlier stays valid but is no longer in the current answer.
//!
//! Membership at a past block handles scheduled removals (the effective block is exclusive),
//! cancellations, and re-registration using the contract's own lifecycle semantics rather than
//! re-deriving them here.
//!
//! Both lookups are historical `eth_call`s at an explicit block, so the endpoint must serve archive
//! state (every fleet node runs `--pruning archive`; third-party operators must too). An RPC that
//! cannot serve the requested state is an error, never an empty or negative answer: treating a
//! refused historical call as "not authorized" would drop finalized messages without a trace.
//! Callers fail closed and retry without advancing their cursor — see
//! [`crate::events::watch_outbox`].

use alloy::eips::BlockNumberOrTag;
use alloy::primitives::Address;
use alloy::providers::Provider;
use anyhow::{Context, Result};

use crate::events::factory::IChainInfo;
use write_ability::abi::IOutboxDiscovery;

/// `chain-info` precompile address (`0x…0fD3`, 4051) on Creditcoin L1 — a runtime precompile
/// registered at `AddressU64<4051>` in creditcoin3 `runtime/src/precompiles.rs`, exposing
/// `pallet_supported_chains::OutboxDiscoveries` (`chain_key → discovery-registry address`) to the
/// EVM. Hand-synced with creditcoin3: it is not an asc-contracts artifact, so the `abi_surface`
/// drift gate cannot check this binding.
pub const CHAIN_INFO_PRECOMPILE: Address = Address::new([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0f, 0xd3,
]);

/// The registry contract the chain key was governed by at one particular source block.
///
/// Resolved per block rather than cached from the head: `set_outbox_discovery_addr` can re-point a
/// chain key at a different registry, and messages finalized under the old one stay authorized
/// there.
pub struct DiscoveryAt {
    /// Registry address governing `chain_key` at the queried block.
    pub address: Address,
}

/// The `uint32` the registry keys Outboxes by, from a route's `u64` `chain_key`.
///
/// A plain `as u32` would wrap 2^32 to 0 and read the registry for a *different* chain, then bind
/// whatever Outbox that answered with — a silent mis-binding. Unrepresentable keys are rejected
/// before any call is made.
pub fn registry_chain_key(chain_key: u64) -> Result<u32> {
    u32::try_from(chain_key).with_context(|| {
        format!(
            "chain_key {chain_key} exceeds the uint32 the Outbox registry is keyed by, so it \
             cannot be represented on-chain"
        )
    })
}

/// Find the governance-registered Discovery for `chain_key` as of `block`.
///
/// Returns `Ok(None)` — not an error — when nothing was registered at that block, which is the
/// normal answer for a chain key whose registry was deployed later. Any RPC failure propagates:
/// an unreachable precompile is not evidence that the registry was absent.
pub async fn discovery_at<P: Provider>(
    provider: &P,
    chain_key: u64,
    block: u64,
) -> Result<Option<DiscoveryAt>> {
    registry_chain_key(chain_key)?;

    let discovery = IChainInfo::new(CHAIN_INFO_PRECOMPILE, provider)
        .get_outbox_discovery_address(chain_key)
        .block(BlockNumberOrTag::Number(block).into())
        .call()
        .await
        .with_context(|| {
            format!(
                "chain_key {chain_key}: historical get_outbox_discovery_address at block {block} \
                 failed — the endpoint must serve archive state (--pruning archive) or this lookup \
                 cannot distinguish an unregistered chain key from an RPC error"
            )
        })?;

    Ok(
        (discovery.exists && !discovery.discoveryAddr.is_zero()).then_some(DiscoveryAt {
            address: discovery.discoveryAddr,
        }),
    )
}

/// Whether `outbox` was an authorized publisher for `chain_key` at `block` under `discovery`.
///
/// This is the authorization check for one candidate Outbox, evaluated against the registry that
/// governed the chain key *at that block*. False means the Outbox was not registered yet, its
/// removal had already become effective, its registration had been cancelled, or a different
/// registry governed the key — all correctly "not authorized here".
///
/// An RPC error is never `false`. Historical state is required for this answer to mean anything,
/// so an endpoint that refuses the call must surface as an error and the caller must retry without
/// advancing its cursor, rather than silently discarding finalized messages.
pub async fn authorized_at<P: Provider>(
    provider: &P,
    chain_key: u64,
    discovery: Address,
    outbox: Address,
    block: u64,
) -> Result<bool> {
    let chain_key_u32 = registry_chain_key(chain_key)?;
    let registry = IOutboxDiscovery::new(discovery, provider);

    registry
        .isActiveOutbox(chain_key_u32, outbox)
        .block(BlockNumberOrTag::Number(block).into())
        .call()
        .await
        .with_context(|| {
            format!(
                "chain_key {chain_key}: historical isActiveOutbox({outbox}) on registry {discovery} \
                 at block {block} failed — refusing to treat an unserved historical call as \
                 unauthorized"
            )
        })
}

/// Every Outbox currently registered as active for `chain_key`.
///
/// The candidate set to authorize against, not the answer on its own: this is the *head* state, so
/// it can miss an Outbox that has since been removed but whose messages are still finalizable. Use
/// it to discover candidates, then confirm each with [`authorized_at`] at the message's block.
///
/// Due-removed entries are already filtered out by the contract. An empty list means the chain key
/// has no Outbox registered right now.
pub async fn active_outboxes<P: Provider>(
    provider: &P,
    chain_key: u64,
    discovery: Address,
) -> Result<Vec<Address>> {
    let chain_key_u32 = registry_chain_key(chain_key)?;
    let registry = IOutboxDiscovery::new(discovery, provider);

    let outboxes = registry
        .activeOutboxes(chain_key_u32)
        .call()
        .await
        .with_context(|| {
            format!("chain_key {chain_key}: activeOutboxes on registry {discovery} failed")
        })?;

    Ok(outboxes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_info_precompile_address_matches_creditcoin3_registration() {
        // AddressU64<4051> per creditcoin3 runtime/src/precompiles.rs: the low 8 bytes of the
        // 20-byte address hold the u64 value big-endian, the rest zero. 4051 decimal = 0xfd3.
        let mut bytes = [0u8; 20];
        bytes[12..20].copy_from_slice(&4051u64.to_be_bytes());
        assert_eq!(CHAIN_INFO_PRECOMPILE, Address::from(bytes));
    }

    /// The contract keys its Outbox maps by `uint32` while routes carry `chain_key` as `u64`. A
    /// plain `as u32` would wrap 2^32 to 0 and read the registry for a *different* chain, then bind
    /// whatever Outbox that answered with. It must refuse instead, and refuse before any call.
    #[test]
    fn refuses_a_chain_key_wider_than_uint32() {
        for key in [u64::from(u32::MAX) + 1, u64::MAX] {
            let err = registry_chain_key(key).unwrap_err();
            let msg = err.to_string();
            assert!(msg.contains("uint32"), "key {key}: {msg}");
        }
    }

    /// The widest representable key must pass: otherwise the guard is off by one and quietly
    /// rejects a legitimate chain key.
    #[test]
    fn accepts_the_largest_representable_chain_key() {
        assert_eq!(registry_chain_key(u64::from(u32::MAX)).unwrap(), u32::MAX);
        assert_eq!(registry_chain_key(0).unwrap(), 0);
    }
}
