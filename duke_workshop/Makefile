
all: markdown
	git add Makefile

MARKDOWN_FOLDER = markdown/

markdown:
	@echo "shit"
	jupyter nbconvert --to markdown ./*.ipynb
	mv ./*.md $(MARKDOWN_FOLDER)
	git add $(MARKDOWN_FOLDER)/*.md

clean:
	-rm $(MARKDOWN_FOLDER)/*.md
