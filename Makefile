SHELL := /bin/bash
REPO_ROOT := $(shell cd .. && pwd)
MANIFEST  := tests/fixtures/legacy_manifest.txt

.PHONY: help
help:
	@echo "PoLiMa - policy framework for SiMa Modalix"
	@echo ""
	@echo "  make install              editable install into the current env"
	@echo "  make test                 unit tests"
	@echo "  make doctor               environment diagnosis"
	@echo "  make check-legacy-intact  assert the 4 legacy stacks are byte-identical"
	@echo "  make backup-legacy        snapshot every legacy script (tree + tarball)"
	@echo "  make list-backups         show available backups"
	@echo "  make restore-legacy       restore the newest backup over the legacy stacks"
	@echo "  make snapshot-legacy      re-record the legacy manifest (deliberate only)"
	@echo "  make lint                 ruff"

.PHONY: install
install:
	pip install -e ".[dev]"

.PHONY: test
test:
	pytest -q tests/unit

.PHONY: doctor
doctor:
	polima doctor

.PHONY: lint
lint:
	ruff check src tests

# The Phase-1 regression gate. Until `git init` lands in Phase 2 this manifest is
# the only proof that PoLiMa has not touched ACT/, SmolVLA/, GR00T-N1.6/ or
# lerobot_sima/. It must pass at every phase boundary.
.PHONY: check-legacy-intact
check-legacy-intact:
	@cd $(REPO_ROOT) && sha256sum -c --quiet polima/$(MANIFEST) \
		&& echo "legacy intact: $$(wc -l < polima/$(MANIFEST)) files unchanged" \
		|| { echo "LEGACY MODIFIED - see above"; exit 1; }

# The manifest DETECTS damage; these targets UNDO it. Backups live in
# MLSandbox/.legacy-backups/<timestamp>/ -- deliberately outside polima/ so that
# blowing away the framework never takes the safety net with it. Each holds a
# browsable tree copy (diff individual scripts directly) plus a tarball.
BACKUP_ROOT := $(REPO_ROOT)/.legacy-backups

# Each backup records its OWN manifest, hashed from the files as copied. It must
# not reuse polima/$(MANIFEST): that describes the last *baseline*, so any
# legitimate edit to a legacy script would make backups fail precisely when one
# is most wanted. Use `make snapshot-legacy` to move the baseline, separately and
# deliberately.
.PHONY: backup-legacy
backup-legacy:
	@cd $(REPO_ROOT) && stamp=$$(date +%Y%m%d_%H%M%S) && dest=".legacy-backups/$$stamp" && \
		mkdir -p "$$dest/tree" && \
		find ACT SmolVLA GR00T-N1.6 lerobot_sima -type f \
			-not -path 'ACT/lerobot/*' -not -path 'SmolVLA/lerobot/*' \
			-not -path 'GR00T-N1.6/Isaac-GR00T/*' \
			-not -path '*/outputs/*' -not -path '*/logs/*' -not -path '*/__pycache__/*' \
			-not -path '*/build/*' -not -name '*.pyc' | sort > "$$dest/files.txt" && \
		rsync -a --files-from="$$dest/files.txt" ./ "$$dest/tree/" && \
		( cd "$$dest/tree" && xargs -a "../files.txt" sha256sum > ../manifest.sha256 ) && \
		tar -czf "$$dest/legacy-first-party-$$stamp.tar.gz" -C "$$dest/tree" . && \
		ln -sfn "$$stamp" .legacy-backups/latest && \
		( cd "$$dest/tree" && sha256sum -c --quiet ../manifest.sha256 ) && \
		printf "backed up %s files -> %s (verified)\n" \
			"$$(wc -l < "$$dest/files.txt")" "$$dest"; \
		if ! diff -q "$$dest/manifest.sha256" polima/$(MANIFEST) >/dev/null 2>&1; then \
			echo "note: this backup differs from the current baseline;"; \
			echo "      run 'make snapshot-legacy' if the change was intended."; \
		fi

.PHONY: list-backups
list-backups:
	@ls -1 $(BACKUP_ROOT) 2>/dev/null | grep -v '^latest$$' | sort || echo "no backups yet"
	@echo "latest -> $$(readlink $(BACKUP_ROOT)/latest 2>/dev/null || echo none)"

# Restores over ACT/ SmolVLA/ GR00T-N1.6/ lerobot_sima/. Only touches files the
# backup contains; never deletes anything. BACKUP=<stamp> selects a specific one.
BACKUP ?= latest
.PHONY: restore-legacy
restore-legacy:
	@test -d "$(BACKUP_ROOT)/$(BACKUP)/tree" || { echo "no backup '$(BACKUP)'"; exit 1; }
	@cd $(REPO_ROOT) && rsync -a "$(BACKUP_ROOT)/$(BACKUP)/tree/" ./ && \
		sha256sum -c --quiet "$(BACKUP_ROOT)/$(BACKUP)/manifest.sha256" && \
		echo "restored from $(BACKUP) and verified"

.PHONY: snapshot-legacy
snapshot-legacy:
	@cd $(REPO_ROOT) && find ACT SmolVLA GR00T-N1.6 lerobot_sima -type f \
		-not -path 'ACT/lerobot/*' -not -path 'SmolVLA/lerobot/*' \
		-not -path 'GR00T-N1.6/Isaac-GR00T/*' \
		-not -path '*/outputs/*' -not -path '*/logs/*' -not -path '*/__pycache__/*' \
		-not -path '*/build/*' -not -name '*.pyc' \
		| sort | xargs sha256sum > polima/$(MANIFEST)
	@echo "recorded $$(wc -l < $(MANIFEST)) files"
