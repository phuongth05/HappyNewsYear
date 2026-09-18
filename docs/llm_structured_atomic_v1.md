# LLM structured atomic evidence extractor

llm_structured_atomic_v1 sends exactly one frozen selected source sentence per
request and constrains the response with the versioned atomic-proposition JSON
schema. Validated proposition content is cached independently from
sample-specific provenance.

For gpt-5.6-terra, the request omits the temperature field because this model
accepts only its provider default. Runs record:

- requested_temperature: null
- effective_temperature: provider_default

The extractor is structurally constrained and cached, but the underlying model
sampling is not guaranteed to be deterministic. Repeated uncached calls may
produce different propositions. Cache hits provide repeatable reuse of a
previously validated response; they do not establish deterministic model
behavior.

Other OpenAI-compatible providers or models may opt into an explicit
temperature through runtime configuration. Temperature is part of the
extraction-content cache key, with omission represented as provider_default, so
explicit and provider-default sampling configurations cannot share cached
content.
