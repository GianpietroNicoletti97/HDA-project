# HDA-project

### Setting up a local repository

The following instructions need to be followed any time a new local repository is created. If you are working in a location where such repo already exists, what follows doesn't need to be repeated every time.

Generate your Token on GitHub following this [guide](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token). Copy the token to a safe place. Remember you can see it only the first time, otherwise if you lose it you have to create another one.

   * Clone your repository (i.e. create a local repository cloned from your remote repository)

   `git clone https://<YourToken>@github.com/GianpietroNicoletti97/HDA-project.git`

   where <YourToken> is the token as copied from the GitHub webpage. A new directory will appear in your current directory. Get into it:

   `cd HDA-project/`

   * Configure your username and email:

   `git config --global user.name "<YourUsername>"`

   `git config --global user.email "<YourEmail>"`

  ### Usual development cycle
  
  Open a command promp in the repository of the project
  `git pull` in order to update the repository.
  
  Edit your code.
  
  `git status` to check the files that was modified.
  
  `git add .` to add all the file or, better, `git add <filename>` (one for each file to add).
  
  `git status` to check the files added.
  
  `git commit -m "message"` to add a message to the file to push.
  
  `git push` to update the remote repository.
  
