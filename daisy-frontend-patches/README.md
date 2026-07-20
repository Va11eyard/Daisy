# Daisytalk/Daisy — legal entity + pricing URL patches

Apply to https://github.com/Daisytalk/Daisy.git (requires Va11eyard push access):

```bash
git clone https://github.com/Daisytalk/Daisy.git
cd Daisy
git checkout -b cursor/legal-entity-pricing-b8a0
git am /path/to/0001-*.patch /path/to/0002-*.patch
git push -u origin cursor/legal-entity-pricing-b8a0
# or: git push origin cursor/legal-entity-pricing-b8a0:main
```

Or from bundle:

```bash
git pull ../daisy-legal-pricing.bundle cursor/legal-entity-pricing-b8a0
```

Changes:
- Daisy Mental Health LLP / BIN 250240028245 → Daisy Labs LLP / BIN 260740027366 (EN + RU legal pages)
- Direct pricing URLs: https://talktodaisy.com/en/pricing and /ru/pricing in terms, privacy, refund
- Nav/footer: `/pricing` instead of `/#pricing`
