# Temp directories
TEMP_DIRS=data/tmp logs results _research/tmp

# Clear temporary files
.PHONY: clear
clear:
	@echo "Clearing temporary files..."
	@rm -rf $(TEMP_DIRS)/*
	@echo "Temporary files cleared."
	@make setup

# Create necessary folders if they do not exist
.PHONY: setup
setup:
	@mkdir -p $(TEMP_DIRS)
	@echo "Directories created: $(TEMP_DIRS)"
