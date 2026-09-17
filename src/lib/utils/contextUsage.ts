export type ContextUsageSnapshot = {
	tokens: number;
	threshold: number | null;
	soft_threshold: number | null;
	source: 'estimated' | 'usage';
};

export type ContextUsageLiveState = 'prefetching' | 'ready' | 'compacting' | 'failed';

export type ContextUsageRingState = 'red' | 'amber' | 'blue' | 'green' | 'normal';

export const isContextUsageSnapshot = (value: unknown): value is ContextUsageSnapshot =>
	typeof value === 'object' &&
	value !== null &&
	typeof (value as ContextUsageSnapshot).tokens === 'number' &&
	Number.isFinite((value as ContextUsageSnapshot).tokens) &&
	((value as ContextUsageSnapshot).threshold === null ||
		typeof (value as ContextUsageSnapshot).threshold === 'number') &&
	((value as ContextUsageSnapshot).soft_threshold === null ||
		typeof (value as ContextUsageSnapshot).soft_threshold === 'number') &&
	(typeof (value as ContextUsageSnapshot).source === 'undefined' ||
		(value as ContextUsageSnapshot).source === 'estimated' ||
		(value as ContextUsageSnapshot).source === 'usage');

export type BranchContextUsage = {
	snapshot: ContextUsageSnapshot;
	messageId: string;
};

type BranchMessage = {
	parentId?: string | null;
	context_usage?: unknown;
};

export const resolveBranchContextUsage = (
	messages: Record<string, BranchMessage> | null | undefined,
	currentId: string | null | undefined
): BranchContextUsage | null => {
	let currentIdWalker: string | null = currentId ?? null;
	while (typeof currentIdWalker === 'string' && currentIdWalker !== '') {
		const message = messages?.[currentIdWalker];
		if (!message) {
			break;
		}
		if (isContextUsageSnapshot(message.context_usage)) {
			return { snapshot: message.context_usage, messageId: currentIdWalker };
		}
		currentIdWalker = message.parentId ?? null;
	}
	return null;
};

export const resolveBranchContextLiveState = (
	messages: Record<string, BranchMessage> | null | undefined,
	currentId: string | null | undefined,
	liveStates: Record<string, ContextUsageLiveState> | null | undefined,
	snapshotOwnerId: string | null | undefined
): ContextUsageLiveState | null => {
	if (!snapshotOwnerId) {
		return null;
	}
	let currentIdWalker: string | null = currentId ?? null;
	while (typeof currentIdWalker === 'string' && currentIdWalker !== '') {
		const state = liveStates?.[currentIdWalker];
		if (state) {
			return state;
		}
		if (currentIdWalker === snapshotOwnerId) {
			return null;
		}
		currentIdWalker = messages?.[currentIdWalker]?.parentId ?? null;
	}
	return null;
};

export const contextUsagePercent = (
	snapshot: ContextUsageSnapshot | null | undefined
): number | null => {
	const threshold = Number(snapshot?.threshold);
	if (!snapshot || !Number.isFinite(threshold) || threshold <= 0) {
		return null;
	}
	return Math.max(0, Math.round((snapshot.tokens / threshold) * 100));
};

export const contextRingState = (
	snapshot: ContextUsageSnapshot | null | undefined,
	liveState: ContextUsageLiveState | null | undefined,
	contextCompactionEnabled: boolean
): ContextUsageRingState | null => {
	if (!contextCompactionEnabled) {
		return null;
	}
	const threshold = Number(snapshot?.threshold);
	if (!snapshot || !Number.isFinite(threshold) || threshold <= 0) {
		return null;
	}
	if (liveState === 'failed') {
		return 'red';
	}
	if (snapshot.tokens > threshold || liveState === 'compacting') {
		return 'amber';
	}
	if (liveState === 'prefetching') {
		return 'blue';
	}
	if (liveState === 'ready') {
		return 'green';
	}
	return 'normal';
};

export const softMarkerPosition = (fraction: number): { x: number; y: number } => ({
	x: 10 + 8 * Math.cos(fraction * 2 * Math.PI),
	y: 10 + 8 * Math.sin(fraction * 2 * Math.PI)
});

type CompactionStatus = {
	action?: unknown;
	description?: unknown;
	done?: unknown;
	error?: unknown;
	phase?: unknown;
	summary?: unknown;
	context_usage?: unknown;
};

const isCompactionStatus = (value: unknown): value is CompactionStatus =>
	typeof value === 'object' &&
	value !== null &&
	(value as CompactionStatus).action === 'context_compaction';

export const applyContextCompactionEvent = (
	liveStates: Record<string, ContextUsageLiveState>,
	readySummaries: Record<string, string>,
	status: unknown,
	messageId: string
): void => {
	if (!isCompactionStatus(status)) {
		return;
	}
	if (status.description === undefined) {
		// Only a freshly accepted send estimate retires a stale blocking
		// failure; a measured usage value keeps the failure on screen.
		if (
			liveStates[messageId] === 'failed' &&
			isContextUsageSnapshot(status.context_usage) &&
			status.context_usage.source === 'estimated'
		) {
			delete liveStates[messageId];
		}
		return;
	}
	if (status.phase === 'prefetch') {
		if (liveStates[messageId] === 'failed' || liveStates[messageId] === 'compacting') {
			// Background prefetch must not visually override blocking compaction.
			if (status.done && typeof status.summary === 'string') {
				readySummaries[messageId] = status.summary;
			}
			return;
		}
		if (status.done) {
			if (status.error) {
				// A failed prefetch stops generation without failing the chat.
				delete liveStates[messageId];
			} else {
				liveStates[messageId] = 'ready';
				if (typeof status.summary === 'string') {
					readySummaries[messageId] = status.summary;
				}
			}
		} else {
			liveStates[messageId] = 'prefetching';
		}
	} else if (status.done) {
		if (status.error) {
			liveStates[messageId] = 'failed';
		} else {
			delete liveStates[messageId];
		}
	} else {
		liveStates[messageId] = 'compacting';
	}
};

export const directContextSummary = (
	message: Record<string, unknown> | null | undefined
): string | null => {
	const value = message?.contextSummary ?? message?.context_summary;
	return typeof value === 'string' && value.trim() ? value : null;
};
