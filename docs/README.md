# docs/ — GitHub Pages site

This directory is the source for the public legal pages of the FairwayCut iOS app, served via GitHub Pages.

It is intentionally separate from the open-source command-line tool in `src/` and contains only static HTML and CSS — no application source.

## Layout

- `index.html` — landing page
- `privacy/index.html` — Privacy Policy
- `terms/index.html` — Terms of Use
- `styles.css` — shared stylesheet
- `.nojekyll` — disables Jekyll processing so these files are served as-is

## Enabling Pages

After this branch is merged to `main`:

1. Repo **Settings** → **Pages**
2. Source: **Deploy from a branch**
3. Branch: **main** / **/docs**
4. Save

The site will be published at `https://itspalomo.github.io/fairwaycut/`.

## Custom domain (optional)

To serve at `https://fairwaycut.app/`, add a `CNAME` file containing `fairwaycut.app` to this directory and configure DNS per [GitHub's custom-domain docs](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site).

## Editing the legal pages

Before publishing, replace the bracketed placeholders in `privacy/index.html` and `terms/index.html`:

- `[LEGAL NAME / DBA]`
- `[STATE]`, `[COUNTY]` (governing-law jurisdiction)

Confirm the contact email and the effective date are correct.
