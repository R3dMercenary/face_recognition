venv: .venv

.venv:
	python3.12 -m venv .venv

install: .venv
	.venv/bin/pip install -r requirements.txt

clean:
	rm -rf .venv

app: install 
	@echo "Starting application..."
	.venv/bin/python app.py

help:
	@echo "Available targets:"
	@echo "  venv    Create virtual environment in .venv"
	@echo "  install Install dependencies into virtual environment"
	@echo "  clean   Remove virtual environment"
	@echo "  help    Show this help message"

.PHONY: venv install clean help