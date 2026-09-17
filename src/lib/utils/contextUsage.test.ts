import { describe, expect, it } from 'vitest';

import {
	applyContextCompactionEvent,
	contextRingState,
	contextUsagePercent,
	directContextSummary,
	isContextUsageSnapshot,
	resolveBranchContextLiveState,
	resolveBranchContextUsage,
	softMarkerPosition,
	type ContextUsageLiveState,
	type ContextUsageSnapshot
} from './contextUsage';

const snapshot = (overrides: Partial<ContextUsageSnapshot> = {}): ContextUsageSnapshot => ({
	tokens: 40,
	threshold: 100,
	soft_threshold: 50,
	source: 'estimated',
	...overrides
});

describe('isContextUsageSnapshot', () => {
	it('accepts the backend snapshot shape', () => {
		expect(isContextUsageSnapshot(snapshot())).toBe(true);
		expect(
			isContextUsageSnapshot(snapshot({ threshold: null, soft_threshold: null, source: 'usage' }))
		).toBe(true);
	});

	it('rejects malformed values', () => {
		expect(isContextUsageSnapshot(null)).toBe(false);
		expect(isContextUsageSnapshot(undefined)).toBe(false);
		expect(isContextUsageSnapshot({ tokens: 0 })).toBe(false);
		expect(isContextUsageSnapshot({ tokens: '40', threshold: 100 })).toBe(false);
		expect(isContextUsageSnapshot({ tokens: 40, threshold: '100' })).toBe(false);
		expect(isContextUsageSnapshot({ tokens: 40, threshold: 100, source: 'other' })).toBe(false);
	});
});

describe('resolveBranchContextUsage', () => {
	const messages = {
		u1: { parentId: null },
		a1: { parentId: 'u1', context_usage: snapshot({ tokens: 10 }) },
		u2: { parentId: 'a1' },
		a2: { parentId: 'u2', context_usage: snapshot({ tokens: 60, source: 'usage' }) },
		sibling: { parentId: 'u2', context_usage: snapshot({ tokens: 999 }) }
	};

	it('returns the nearest snapshot on the selected branch', () => {
		const resolved = resolveBranchContextUsage(messages as any, 'a2');
		expect(resolved?.messageId).toBe('a2');
		expect(resolved?.snapshot.tokens).toBe(60);
		expect(resolved?.snapshot.source).toBe('usage');
	});

	it('walks ancestors when the tip has no snapshot', () => {
		const tip = { parentId: 'a2' };
		const resolved = resolveBranchContextUsage({ ...messages, tip: tip as any } as any, 'tip');
		expect(resolved?.messageId).toBe('a2');
		expect(resolved?.snapshot.tokens).toBe(60);
	});

	it('never borrows from sibling responses', () => {
		expect(resolveBranchContextUsage(messages as any, 'sibling')?.snapshot.tokens).toBe(999);
		const withoutOwn = resolveBranchContextUsage(
			{ ...messages, sibling: { parentId: 'u2' } } as any,
			'sibling'
		);
		expect(withoutOwn?.snapshot.tokens).toBe(10);
	});

	it('treats missing snapshots as unmeasured, not zero', () => {
		expect(resolveBranchContextUsage({ u1: { parentId: null } } as any, 'u1')).toBeNull();
		expect(resolveBranchContextUsage(messages as any, null)).toBeNull();
		expect(resolveBranchContextUsage(null, 'a2')).toBeNull();
		expect(resolveBranchContextUsage(messages as any, 'missing')).toBeNull();
	});
});

describe('resolveBranchContextLiveState', () => {
	const messages = {
		u1: { parentId: null },
		parent: { parentId: 'u1', context_usage: snapshot({ tokens: 40 }) },
		child: { parentId: 'parent' },
		otherTip: { parentId: 'parent' }
	};

	it('uses the nearest state on the path from the tip to the snapshot owner', () => {
		const liveStates: Record<string, ContextUsageLiveState> = {
			child: 'failed',
			parent: 'ready'
		};
		expect(resolveBranchContextLiveState(messages as any, 'child', liveStates, 'parent')).toBe(
			'failed'
		);
		expect(resolveBranchContextLiveState(messages as any, 'parent', liveStates, 'parent')).toBe(
			'ready'
		);
	});

	it('falls back to the owner state when the tip has none', () => {
		expect(
			resolveBranchContextLiveState(messages as any, 'child', { parent: 'compacting' }, 'parent')
		).toBe('compacting');
		expect(resolveBranchContextLiveState(messages as any, 'child', {}, 'parent')).toBeNull();
	});

	it('never borrows a sibling state', () => {
		expect(
			resolveBranchContextLiveState(messages as any, 'child', { otherTip: 'failed' }, 'parent')
		).toBeNull();
	});

	it('returns null without a snapshot owner', () => {
		expect(
			resolveBranchContextLiveState(messages as any, 'child', { child: 'failed' }, null)
		).toBeNull();
	});
});

describe('contextUsagePercent', () => {
	it('derives the percent from tokens and threshold', () => {
		expect(contextUsagePercent(snapshot({ tokens: 40, threshold: 100 }))).toBe(40);
		expect(contextUsagePercent(snapshot({ tokens: 250, threshold: 100 }))).toBe(250);
	});

	it('returns null without a threshold', () => {
		expect(contextUsagePercent(snapshot({ threshold: null }))).toBeNull();
		expect(contextUsagePercent(null)).toBeNull();
	});
});

describe('contextRingState', () => {
	it('hides the ring when compaction is disabled or the snapshot lacks a threshold', () => {
		expect(contextRingState(snapshot(), null, false)).toBeNull();
		expect(contextRingState(null, null, true)).toBeNull();
		expect(contextRingState(snapshot({ threshold: null }), 'prefetching', true)).toBeNull();
	});

	it('does not treat the rounded boundary as over-threshold', () => {
		expect(contextRingState(snapshot({ tokens: 99_500, threshold: 100_000 }), null, true)).toBe(
			'normal'
		);
		expect(contextRingState(snapshot({ tokens: 100_000, threshold: 100_000 }), null, true)).toBe(
			'normal'
		);
		expect(contextRingState(snapshot({ tokens: 100_001, threshold: 100_000 }), null, true)).toBe(
			'amber'
		);
	});

	it('applies the red over amber over blue over green over normal priority', () => {
		expect(contextRingState(snapshot({ tokens: 40 }), 'failed', true)).toBe('red');
		expect(contextRingState(snapshot({ tokens: 120 }), 'failed', true)).toBe('red');
		expect(contextRingState(snapshot({ tokens: 120 }), 'compacting', true)).toBe('amber');
		expect(contextRingState(snapshot({ tokens: 120 }), null, true)).toBe('amber');
		expect(contextRingState(snapshot({ tokens: 40 }), 'compacting', true)).toBe('amber');
		expect(contextRingState(snapshot({ tokens: 40 }), 'prefetching', true)).toBe('blue');
		expect(contextRingState(snapshot({ tokens: 120 }), 'prefetching', true)).toBe('amber');
		expect(contextRingState(snapshot({ tokens: 40 }), 'ready', true)).toBe('green');
		expect(contextRingState(snapshot({ tokens: 40 }), null, true)).toBe('normal');
	});

	it('does not turn blue for merely crossing the soft threshold', () => {
		expect(contextRingState(snapshot({ tokens: 55, soft_threshold: 50 }), null, true)).toBe(
			'normal'
		);
	});
});

describe('softMarkerPosition', () => {
	it('places the marker at 6 oclock for a half threshold under the rotated ring', () => {
		const { x, y } = softMarkerPosition(0.5);
		expect(x).toBeCloseTo(2, 6);
		expect(y).toBeCloseTo(10, 6);
	});

	it('starts at the arc origin for a zero fraction', () => {
		const { x, y } = softMarkerPosition(0);
		expect(x).toBeCloseTo(18, 6);
		expect(y).toBeCloseTo(10, 6);
	});
});

describe('applyContextCompactionEvent', () => {
	it('clears the generating state after a failed prefetch without touching others', () => {
		const liveStates: Record<string, any> = {};
		const readySummaries: Record<string, string> = {};
		applyContextCompactionEvent(
			liveStates,
			readySummaries,
			{
				action: 'context_compaction',
				description: 'Generating summary in advance',
				done: false,
				phase: 'prefetch'
			},
			'm1'
		);
		expect(liveStates['m1']).toBe('prefetching');
		applyContextCompactionEvent(
			liveStates,
			readySummaries,
			{
				action: 'context_compaction',
				description: 'Context compaction failed',
				done: true,
				error: true,
				phase: 'prefetch'
			},
			'm1'
		);
		expect(liveStates['m1']).toBeUndefined();
		expect(readySummaries['m1']).toBeUndefined();
	});

	it('records ready summaries and clears them on blocking completion', () => {
		const liveStates: Record<string, any> = {};
		const readySummaries: Record<string, string> = {};
		applyContextCompactionEvent(
			liveStates,
			readySummaries,
			{
				action: 'context_compaction',
				description: 'Summary ready',
				done: true,
				phase: 'prefetch',
				summary: 'S1'
			},
			'm1'
		);
		expect(liveStates['m1']).toBe('ready');
		expect(readySummaries['m1']).toBe('S1');
		applyContextCompactionEvent(
			liveStates,
			readySummaries,
			{ action: 'context_compaction', description: 'Context compacted', done: true },
			'm1'
		);
		expect(liveStates['m1']).toBeUndefined();
	});

	it('keeps failed and compacting live states across prefetch events', () => {
		const prefetchEvents = [
			{
				action: 'context_compaction',
				description: 'Generating summary in advance',
				done: false,
				phase: 'prefetch'
			},
			{
				action: 'context_compaction',
				description: 'Summary ready',
				done: true,
				phase: 'prefetch',
				summary: 'S1'
			},
			{
				action: 'context_compaction',
				description: 'Context compaction failed',
				done: true,
				error: true,
				phase: 'prefetch'
			}
		];
		for (const protectedState of ['failed', 'compacting'] as const) {
			for (const event of prefetchEvents) {
				const liveStates: Record<string, ContextUsageLiveState> = { m1: protectedState };
				const readySummaries: Record<string, string> = {};
				applyContextCompactionEvent(liveStates, readySummaries, event, 'm1');
				expect(liveStates['m1']).toBe(protectedState);
			}
		}
	});

	it('still records the success summary while a live state is protected', () => {
		const liveStates: Record<string, ContextUsageLiveState> = { m1: 'compacting' };
		const readySummaries: Record<string, string> = {};
		applyContextCompactionEvent(
			liveStates,
			readySummaries,
			{
				action: 'context_compaction',
				description: 'Summary ready',
				done: true,
				phase: 'prefetch',
				summary: 'S1'
			},
			'm1'
		);
		expect(liveStates['m1']).toBe('compacting');
		expect(readySummaries['m1']).toBe('S1');
	});

	it('tracks blocking compaction and keeps failures red', () => {
		const liveStates: Record<string, any> = {};
		applyContextCompactionEvent(
			liveStates,
			{},
			{ action: 'context_compaction', description: 'Compacting context', done: false },
			'm1'
		);
		expect(liveStates['m1']).toBe('compacting');
		applyContextCompactionEvent(
			liveStates,
			{},
			{
				action: 'context_compaction',
				description: 'Context compaction failed',
				done: true,
				error: true
			},
			'm1'
		);
		expect(liveStates['m1']).toBe('failed');
	});

	it('keeps failed after a usage snapshot but retires it after an estimated snapshot', () => {
		const usageStates: Record<string, ContextUsageLiveState> = { m1: 'failed' };
		applyContextCompactionEvent(
			usageStates,
			{},
			{ action: 'context_compaction', context_usage: snapshot({ tokens: 30, source: 'usage' }) },
			'm1'
		);
		expect(usageStates['m1']).toBe('failed');

		const estimateStates: Record<string, ContextUsageLiveState> = { m1: 'failed' };
		applyContextCompactionEvent(
			estimateStates,
			{},
			{ action: 'context_compaction', context_usage: snapshot({ tokens: 40 }) },
			'm1'
		);
		expect(estimateStates['m1']).toBeUndefined();
	});

	it('keeps prefetching, ready, and recorded summaries across estimated snapshots', () => {
		const liveStates: Record<string, ContextUsageLiveState> = {
			prefetching: 'prefetching',
			ready: 'ready'
		};
		const readySummaries: Record<string, string> = { ready: 'S1' };
		applyContextCompactionEvent(
			liveStates,
			readySummaries,
			{ action: 'context_compaction', context_usage: snapshot({ tokens: 40 }) },
			'prefetching'
		);
		applyContextCompactionEvent(
			liveStates,
			readySummaries,
			{ action: 'context_compaction', context_usage: snapshot({ tokens: 40 }) },
			'ready'
		);
		expect(liveStates['prefetching']).toBe('prefetching');
		expect(liveStates['ready']).toBe('ready');
		expect(readySummaries['ready']).toBe('S1');
	});
});

describe('directContextSummary', () => {
	it('reads the message-level summary only', () => {
		expect(directContextSummary({ contextSummary: ' direct ' })).toBe(' direct ');
		expect(directContextSummary({ context_summary: 'snake' })).toBe('snake');
		expect(directContextSummary({ contextSummary: '   ' })).toBeNull();
		expect(
			directContextSummary({
				output: [{ type: 'message', contextSummary: 'nested only' }]
			})
		).toBeNull();
		expect(directContextSummary(null)).toBeNull();
	});
});
