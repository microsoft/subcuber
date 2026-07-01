# Test Makefile for security audit - approval gate verification
# This is a benign file to test if fork PRs require workflow approval

.PHONY: codeql
codeql:
	@echo "Security audit: workflow approval gate test - no code executed"
	@echo "Runner hostname: $$(hostname 2>/dev/null || echo unknown)"
	@echo "Current user: $$(whoami 2>/dev/null || echo unknown)"
	@echo "Working directory: $$(pwd)"
	@echo "Test completed successfully."
