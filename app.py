"""Entry point for the Hugging Face Space. Everything real is in the package.

A Space runs `python app.py` at the repository root and expects a Gradio app to
start listening; that is the whole contract, and this file is the whole of my
half of it. The page is :mod:`filing.gradio_app`, the engine is
:class:`filing.api.AskEngine`, and both are the same code the local API serves
-- a deployment that forked its own logic to fit a host would be demonstrating
the host, not the system.

The Space's configuration is the frontmatter in README.md. Two Space
*variables* (neither is a secret, so neither belongs in the secrets tab):

    QDRANT_PATH=data/qdrant   the embedded store, written by `filing pack`
    TRACING_ENABLED=false     no collector is listening out here, and exporting
                              into a closed port would put a failed connection
                              in the log on every span. The page already has a
                              state for this and says so under the answer: no
                              trace to link to, rather than a dead link.

and one actual secret, `GEMINI_API_KEY`, which does go in the secrets tab so it
stays out of the repository and out of the process listing.

The configuration the agent runs under is `agent-guarded` by default, which is
the row the ablation table reports and the one whose numbers the README quotes.
"""

from filing.gradio_app import main

if __name__ == "__main__":
    main()
