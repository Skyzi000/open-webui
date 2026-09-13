import { describe, expect, it } from 'vitest';

import {
	buildOutputDisplayItems,
	getOutputText,
	type OutputDisplayItem,
	type OutputItem
} from './structuredOutput';

function messageItem(id: string, text: string): OutputItem {
	return {
		type: 'message',
		id,
		status: 'completed',
		role: 'assistant',
		content: [{ type: 'output_text', text }]
	};
}

function functionCallItem(callId: string, name: string): OutputItem {
	return {
		type: 'function_call',
		id: callId,
		call_id: callId,
		name,
		arguments: '{}',
		status: 'completed'
	};
}

function functionOutputItem(callId: string, text: string): OutputItem {
	return {
		type: 'function_call_output',
		id: `${callId}-out`,
		call_id: callId,
		status: 'completed',
		output: [{ type: 'input_text', text }]
	};
}

function adoptionItem(id: string, summary: string): OutputItem {
	return {
		type: 'open_webui:context_compaction',
		id,
		compaction_summary: summary
	};
}

function shapes(items: OutputDisplayItem[]): string[] {
	return items.map((item) => {
		if (item.type === 'context_compaction') {
			return `${item.variant}:${item.summary}`;
		}
		return item.type;
	});
}

describe('buildOutputDisplayItems compaction boundaries', () => {
	it('places a nested checkpoint divider immediately before its carrier output item', () => {
		const carrier = {
			...functionCallItem('call-a', 'lookup'),
			contextSummary: 'CHECKPOINT'
		};
		const items = buildOutputDisplayItems([
			messageItem('m1', 'before'),
			carrier,
			functionOutputItem('call-a', 'result'),
			messageItem('m2', 'after')
		]);

		expect(shapes(items)).toEqual([
			'message',
			'checkpoint:CHECKPOINT',
			'detail_single',
			'message'
		]);
	});

	it('renders adoption records as dividers instead of normal output', () => {
		const items = buildOutputDisplayItems([
			adoptionItem('cc_1', 'S1'),
			functionCallItem('call-a', 'lookup'),
			functionOutputItem('call-a', 'result'),
			adoptionItem('cc_2', 'S2'),
			messageItem('m1', 'done')
		]);

		expect(shapes(items)).toEqual([
			'adoption:S1',
			'detail_single',
			'adoption:S2',
			'message'
		]);
	});

	it('keeps call/result pairing and reasoning-first outputs across dividers', () => {
		const items = buildOutputDisplayItems([
			{
				type: 'reasoning',
				id: 'r1',
				status: 'completed',
				content: [{ type: 'output_text', text: 'thinking' }]
			},
			functionCallItem('call-a', 'lookup'),
			functionOutputItem('call-a', 'result'),
			adoptionItem('cc_1', 'S1'),
			functionCallItem('call-b', 'fetch'),
			functionOutputItem('call-b', 'result-b'),
			messageItem('m1', 'final')
		]);

		expect(shapes(items)).toEqual(['detail_group', 'adoption:S1', 'detail_single', 'message']);
	});

	it('uses stable ids derived from output item ids so open state survives stream updates', () => {
		const output = [
			adoptionItem('cc_1', 'S1'),
			{
				...functionCallItem('call-a', 'lookup'),
				contextSummary: 'CHECKPOINT'
			},
			functionOutputItem('call-a', 'result'),
			messageItem('m1', 'done')
		];
		const first = buildOutputDisplayItems(output);
		const second = buildOutputDisplayItems([...output]);

		expect(first.map((item) => item.id)).toEqual(second.map((item) => item.id));
		expect(first.map((item) => item.id)).toContain('adoption-cc_1');
		expect(first.map((item) => item.id)).toContain('checkpoint-call-a');
	});

	it('keeps adoption text out of extracted message content', () => {
		const output = [
			adoptionItem('cc_1', 'SUMMARY BODY'),
			{
				...functionCallItem('call-a', 'lookup'),
				contextSummary: 'CHECKPOINT BODY'
			},
			functionOutputItem('call-a', 'result'),
			messageItem('m1', 'answer')
		];

		expect(getOutputText(output)).toBe('answer');
	});

	it('ignores blank checkpoint summaries and non-string adoption summaries', () => {
		const items = buildOutputDisplayItems([
			{ ...functionCallItem('call-a', 'lookup'), contextSummary: '   ' },
			{ ...adoptionItem('cc_1', ''), compaction_summary: 42 },
			messageItem('m1', 'answer')
		]);

		expect(shapes(items)).toEqual(['detail_single', 'message']);
	});
});
