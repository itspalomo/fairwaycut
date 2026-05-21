# docs/ — GitHub Pages site

This directory is the source for the public legal pages of the FairwayCut iOS app, served via GitHub Pages.

It is intentionally separate from the open-source command-line tool in `src/` and contains only static HTML and CSS — no application source. These legal pages apply to the iOS app only. The CLI remains governed by the repository's MIT License.

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

Identifying details currently in use:

- DBA: **itsPalomo**
- Governing law: **California**, venue **Santa Clara County** (Palo Alto)
- Contact email: **itspalomo.dev@gmail.com** (placeholder; rotate to a dedicated address before broad distribution)

Update the effective date when publishing material changes.
