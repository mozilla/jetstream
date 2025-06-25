# Analysis Infrastructure Updates

## Context

- project was largely (if not entirely) configured by hand
- no staging environment
- difficult to test changes to infra
- difficult to manage infra

## Ideas

- infra as code
- staging env (or at least stage cluster)
- exposed Argo UI
  - see exposed ArgoCD from SRE
  - this looks simple but is an example of difficult to test without a risk to prod cluster