
jupyter nbconvert --to markdown ./*.ipynb
mv ./*.md markdown/
git add markdown/*.md
