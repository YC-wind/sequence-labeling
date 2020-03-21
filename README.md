分词与词性标注

1.Checkout

   git checkout --orphan latest_branch

2. Add all the files

   git add -A

3. Commit the changes

   git commit -am "commit message"


4. Delete the branch

   git branch -D master

5.Rename the current branch to master

   git branch -m master

6.Finally, force update your repository

   git push -f origin master
--------------------- 

find . -name .DS_Store -print0 | xargs -0 git rm -f --ignore-unmatch

echo .DS_Store >> ~/.gitignore

git add --all

git commit -m '.DS_Store banished!'


find . -name '*.DS_Store' -type f -delete

