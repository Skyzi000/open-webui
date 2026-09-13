<script lang="ts">
	import { getContext } from 'svelte';
	import type { Writable } from 'svelte/store';
	import type { i18n as i18nType } from 'i18next';

	const i18n = getContext<Writable<i18nType>>('i18n');

	import Collapsible from '$lib/components/common/Collapsible.svelte';

	export let summary = '';
	export let variant: 'checkpoint' | 'adoption' = 'checkpoint';
</script>

<div class="my-1 border-t border-gray-100 dark:border-gray-850"></div>

<Collapsible
	title={variant === 'checkpoint'
		? $i18n.t('Conversation summarized up to here')
		: $i18n.t('Using this summary from here')}
	className="w-full"
	buttonClassName="w-fit py-1 text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition"
>
	<div slot="content" class="mb-2">
		<div class="whitespace-pre-wrap break-words text-xs text-gray-600 dark:text-gray-400">
			{summary}
		</div>
	</div>
</Collapsible>
