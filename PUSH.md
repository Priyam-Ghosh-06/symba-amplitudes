# Pushing this to your GitHub account

Everything is committed locally. What is missing is credentials — I have no way
to authenticate as you, so the last step is yours. It is one command.

## Why I can't do this part

I never see your GitHub password or a token, and I won't ask you to paste one
into a chat. The `gh` CLI isn't installed on this machine and the GitHub
connector isn't authorised in this session, so there is no path for me to create
a repo under your account.

The good news: **Git Credential Manager is already installed** with your Git for
Windows. Once a remote exists, `git push` opens a browser window, you sign in to
GitHub once, and GCM stores the credential in Windows Credential Manager. Every
push after that is automatic — no tokens to manage, nothing to paste anywhere.

## Do this

**1. Create an empty private repo** at <https://github.com/new>

- Name: `symba-amplitudes` (or whatever you prefer)
- Visibility: **Private**
- **Do not** tick "Add a README", ".gitignore", or "license" — the repo must be
  empty or the first push will conflict.

**2. Run this,** with your repo URL:

```bash
cd /d/SYMBA/symba-amplitudes && git remote add origin https://github.com/Priyam-Ghosh-06/symba-amplitudes.git && git push -u origin main
```

A browser window will open the first time. Sign in, approve, done.

If you named the repo something else, swap the URL. If you re-run it and get
`remote origin already exists`, use `git remote set-url origin <url>` instead of
`git remote add`.

## Making it public later

After you have reviewed it: repo → Settings → General → scroll to the bottom →
Danger Zone → Change visibility → Public.

Before you do, two things worth a look:

- `data/Symba/` is committed (808 KB of the QED/QCD corpora). If that data isn't
  yours to redistribute, `git rm -r --cached data/Symba` and add it to
  `.gitignore` before going public.
- `docs/` contains the internal design critique. It is good material and I would
  keep it, but it is candid about what was wrong with the earlier notebooks, so
  it is your call whether that is public-facing.

## Preferring the `gh` CLI instead

If you would rather have one command that creates the repo *and* pushes:

```bash
winget install --id GitHub.cli
```

Then open a new terminal and run:

```bash
gh auth login
```

(pick GitHub.com → HTTPS → login with a web browser), and finally:

```bash
cd /d/SYMBA/symba-amplitudes && gh repo create symba-amplitudes --private --source=. --remote=origin --push
```

Delete this file once the repo is up.
