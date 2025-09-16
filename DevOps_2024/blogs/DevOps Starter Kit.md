
# A Beginner's Guide to DevOps** 🚀 
> **Author: Mitansh Manav** 
-----------------------

Welcome to the world of DevOps! If you're new here, the sheer number of tools and concepts can seem overwhelming. But don't worry, we're here to break down the essentials. This guide will walk you through the core concepts of **Git**, **GitHub**, **Agile**, and more which are fundamental to modern DevOps practices.

## What is DevOps?

DevOps is a philosophy that aims to shorten the software development life cycle and provide continuous delivery with high software quality. It's a blend of cultural philosophies, practices, and tools that increases an organization's ability to deliver applications and services at high velocity. At its heart, it's about **breaking down silos** between development (**Dev**) and operations (**Ops**) teams.

---

## Agile: The Guiding Philosophy

Before we dive into the tools, let's talk about the mindset. **Agile** is an iterative approach to project management and software development that helps teams deliver value to their customers faster and with fewer headaches. Instead of a single, large release at the end of a project (**waterfall**), Agile focuses on delivering small, incremental releases.

* **Sprints**: An Agile team works in short, fixed-length cycles called **sprints**, typically lasting one to four weeks.
* **User Stories**: Features are described from the user's perspective, like "As a user, I want to log in so I can access my account."
* **Stand-ups**: Teams have brief, daily meetings (**stand-ups**) to discuss what they did yesterday, what they'll do today, and any roadblocks they face. This promotes transparency and quick problem-solving.

***Use Case***: Imagine building a simple e-commerce website. A waterfall approach would mean designing everything at once, building for months, and then releasing it. With Agile, you'd release a basic login page first (sprint 1), then add the product catalog (sprint 2), and finally, the shopping cart and checkout (sprint 3). This way, you get a working product to market faster and can gather user feedback along the way.

---

## **Scrum: A Deep Dive**

Scrum is one of the most popular Agile **frameworks.** It provides a structured way to manage work by organizing teams into short, time-boxed cycles.

* **Roles**: Scrum defines specific roles to keep the process on track:
    * **Product Owner**: Defines and prioritizes the work in the product backlog. They are the voice of the customer.
    * **Scrum Master**: A coach and facilitator who helps the team understand and apply Scrum practices. They remove obstacles that block the team's progress.
    * **Development Team**: A cross-functional group responsible for creating the product. They are self-organizing and manage their own work.

* **Events (Ceremonies)**: These are the scheduled meetings that keep the process running smoothly:
    * **Sprint Planning**: At the start of a sprint, the team decides what to work on based on the prioritized backlog.
    * **Daily Scrum (Stand-up)**: A brief daily meeting where the team syncs up on progress and identifies any impediments.
    * **Sprint Review**: A meeting at the end of the sprint to demonstrate the completed work to stakeholders and get feedback.
    * **Sprint Retrospective**: The team reflects on the sprint and identifies what went well and what could be improved for the next one.

***Scrum in Action***: A team is tasked with adding a new payment gateway to their application. The Product Owner creates user stories for the feature and adds them to the product backlog. In Sprint Planning, the team commits to a set of stories they believe they can complete in the next two weeks. Each day, they hold a 15-minute stand-up to discuss progress. At the end of the two weeks, they have a working, tested payment gateway, which they demonstrate in the Sprint Review. The team then holds a Retrospective to discuss how they can make their testing process more efficient in the future.

---

## **Kanban: Continuous Flow in Action**

Kanban is another popular Agile **methodology** that focuses on visualizing work, limiting work in progress (WIP), and maximizing efficiency. Unlike Scrum's time-boxed sprints, Kanban is all about **continuous flow.**

* **The Kanban Board**: The core of Kanban is a visual board that represents the workflow. It's a simple, powerful tool with columns for each stage of the process, such as `To Do`, `In Progress`, and `Done`.
* **Work in Progress (WIP) Limits**: This is the most crucial principle of Kanban. WIP limits restrict the number of tasks that can be "in progress" at any given time. This prevents team members from multitasking and creates a "pull" system where new work is only pulled into the workflow when there is capacity.
* **Continuous Delivery**: With no fixed sprints, new work can be released as soon as it's completed and ready, allowing for faster and more frequent deployments.

***Kanban in Action***: A DevOps team uses a Kanban board to manage their infrastructure requests. The columns are `Backlog`, `Ready to Work`, `Configuring Server`, `Testing`, and `Done`. The team sets a WIP limit of 2 for the `Configuring Server` column. When a developer completes a task and moves it to the `Testing` column, a space opens up, allowing them to "pull" a new task from the `Ready to Work` column. This prevents bottlenecks and ensures that work is always flowing smoothly through the pipeline.

---

## Git: Your Local Version Control System

**Git** is a powerful, distributed **version control system** (VCS). Think of it as a super-smart "undo" button for your code. It tracks changes to your files and allows you to revert to previous versions at any time. The key thing to remember is that Git operates on your local machine.

### Essential Git Commands

You don't need to know every command to get started. Here are the core ones you'll use daily:

1.  **`git clone`**: This command creates a local copy of a remote repository.
    * **Example**: `git clone https://github.com/Mk-CloudLeader/aws_Meetup-2023.git`

2.  **`git status`**: Checks the status of your working directory. It tells you which files are modified, staged for a commit, or untracked.

3.  **`git add`**: Adds changes in your working directory to the **staging area**.
    * **Example**: `git add .` (adds all changed files) or `git add index.html` (adds a specific file)

4.  **`git commit`**: Takes the staged changes and records them in your local repository. You should always include a descriptive message.
    * **Example**: `git commit -m "feat: Add user login feature"`

5.  **`git push`**: Pushes your committed changes from your local repository to a remote repository (like GitHub).
    * **Example**: `git push origin main`

---

## GitHub: Your Collaborative Hub

While Git is a local tool, **GitHub** is a web-based platform that hosts Git repositories. It's where teams collaborate, share code, and manage projects. Think of it as social media for developers.

### The GitHub Workflow

This is a common workflow for working with GitHub:

1.  **Install Git and Create a GitHub Account**: If you haven't already, follow these steps:
    * Install Git: `https://git-scm.com/book/en/v2/Getting-Started-Installing-Git`
    * Create a GitHub Account: `https://github.com/join`

2.  **Clone the Repository**: Get a local copy of the project.
    * **Command**: `git clone [repository URL]`

3.  **Create a New Branch**: Branches are isolated lines of development. You should always create a new branch for your work to avoid directly changing the main code.
    * **Command**: `git checkout -b my-new-feature`

4.  **Make Changes & Commit**: Work on your code, then use `git add` and `git commit` to save your changes locally.

5.  **Push Your Branch**: Share your work with the team on GitHub.
    * **Command**: `git push origin my-new-feature`

6.  **Open a Pull Request (PR)**: A **Pull Request** is a way of notifying others about your changes. It's a request to merge your branch into the main branch. This is where code reviews happen, and other team members can provide feedback.

7.  **Merge the PR**: Once your changes have been reviewed and approved, the PR can be merged into the main branch, making your code part of the project.

***Use Case***: A developer is assigned to fix a bug on a website. They create a new branch called `bugfix/login-issue`, make the fix, commit their changes, and push the branch to GitHub. They then open a PR, and a teammate reviews the code to ensure the fix is correct and doesn't introduce new problems. Once approved, the changes are merged into the `main` branch, and the fix is ready for deployment.

---

## Bringing It All Together

DevOps isn't just about tools; it's about a culture of collaboration and continuous improvement. By adopting Agile practices and mastering tools like Git and GitHub, you're laying the foundation for a successful DevOps journey. You're not just writing code; you're contributing to a shared, transparent, and efficient process. So, get comfortable with these concepts, and you'll be well on your way!

Happy coding! 👩‍💻👨‍💻

## Git Cheatsheet:

<img src="https://github.com/manav-dl/aws_Meetup-2023/blob/de52a887647fe51df5c56108e2380d655c8ae0dc/DevOps_2024/Excercise/Images/git-commands.png" style="width: 800px; height: 720px;">

## GitHub Workflow:

<img src="https://github.com/manav-dl/aws_Meetup-2023/blob/de52a887647fe51df5c56108e2380d655c8ae0dc/DevOps_2024/Excercise/Images/GitHubWorkFlow.svg" style="width: 1080px; height: 720px;">

### Forking a Repository:

***

![forking](https://github.com/manav-dl/aws_Meetup-2023/assets/122433722/7cc60b67-423a-4434-b5b5-d314271da70f)


### Creating the Fork:

***

![creatingFork](https://github.com/manav-dl/aws_Meetup-2023/assets/122433722/d45f2343-3e6e-4fb1-8fac-2d22ee166c05)

### Cloning with https:

***

![CloningRepo](https://github.com/manav-dl/aws_Meetup-2023/assets/122433722/97fcb70b-e5e2-4165-9eb8-bdbc1b389f4a)

### Making a local clone with git clone:

***
- ```git clone <url>```

![gitclone](https://github.com/manav-dl/aws_Meetup-2023/assets/122433722/673883aa-a62d-4c04-b6b9-48f47b7a0197)

### Updating local repo with git pull:

***
- ```git pull```

![gitpull](https://github.com/manav-dl/aws_Meetup-2023/assets/122433722/79612733-2dcb-4db1-a266-fd2ed70aa9db)

### Creating a new branch and switching to it:

***
- Creating a Branch: ```git branch <name>```
- Switching to a Branch: ```git checkout <name>```

![gitbranch](https://github.com/manav-dl/aws_Meetup-2023/assets/122433722/6262f298-97e7-42db-94fc-2ed9b28026d3)

### Adding the file:

***
- ```git add <file name>```

![gitadd](https://github.com/manav-dl/aws_Meetup-2023/assets/122433722/0e4f8f55-9b7f-49ff-9474-23519070ed1f)

### Commiting the change:

***
- ```git commit -m "message"```

![gitcommit](https://github.com/manav-dl/aws_Meetup-2023/assets/122433722/b2324c3e-48d2-4491-ac10-8e1718c38c59)

### Merging the changes:

***
- First switch to main branch with ```git checkout <main branch name>```
- Use ```git merge <branch name>``` to merge it with the main branch

![gitmerge](https://github.com/manav-dl/aws_Meetup-2023/assets/122433722/7595e0a9-2de3-4005-b21a-5bee5e92b8ee)

### Pushing changes to Central Repository:

***
- ```git push```

![git-push](https://github.com/manav-dl/aws_Meetup-2023/assets/122433722/2571f9e2-b52a-469a-b1c6-d4b3ba791e1d)

### Opening a Pull Request:

***
- Step 1:

![pull_req_1](https://github.com/manav-dl/aws_Meetup-2023/assets/122433722/68d163cc-ac25-43d4-bfd6-b6d0b407a55d)

- Step 2:

![pull_request_complete](https://github.com/manav-dl/aws_Meetup-2023/assets/122433722/e292ae3e-f4e9-4857-a7dc-9b51605c832e)

***

***Central Repository's maintainers will review and merge the changes***




# **FAQs: DevOps**

---

### **Git & GitHub**

1.  **What is the difference between Git and GitHub?**
    **Git** is a local, open-source version control system you install on your computer. **GitHub** is a cloud-based hosting service for Git repositories. Git is the tool, and GitHub is the platform that uses Git for collaboration and project management.

2.  **What is a repository?**
    A repository (or "repo") is a central storage location for all the project's files, including code, documentation, and revision history.

3.  **What is a branch?**
    A branch is an isolated, parallel version of your repository. It allows developers to work on new features or bug fixes without affecting the main codebase.

4.  **What is the `git clone` command for?**
    `git clone` is used to create a local copy of a remote Git repository on your machine.

5. **What is a Pull Request (PR)?**
    A Pull Request is a proposal to merge changes from one branch into another. It's a key part of the collaboration workflow on platforms like GitHub, as it allows teammates to review, comment on, and discuss the changes before they are merged.

6. **What is the `git push` command for?**
    `git push` is used to upload your local commits to a remote repository (e.g., GitHub).

7. **How do I undo a change in Git?**
    This depends on the change. You can use `git reset` to undo local commits or changes in the staging area. For pushed commits, you can use `git revert` to create a new commit that undoes the previous one.

---

### **Agile, Scrum & Kanban**

8. **What is Agile?**
    Agile is a set of principles for software development that advocates for adaptive planning, evolutionary development, early delivery, and continuous improvement, and it encourages rapid and flexible response to change.

9. **What is the difference between Scrum and Kanban?**
    * **Scrum** is a framework that uses short, time-boxed iterations called sprints to deliver work. It has specific roles (Product Owner, Scrum Master) and ceremonies (Daily Scrums, Sprint Review).
    * **Kanban** is a methodology that focuses on visualizing work on a board and limiting work-in-progress (WIP). It emphasizes a continuous flow of work rather than fixed cycles.
**Scrum use case** Team is tasked with developing a new "search" functionality for web app.
  - Product team creates and prioritizes a backlog of stories
  - Sprint planning : Team picks wich stories to complete in next 2 weeks
  - Sprint Review : Demo the search feature to stakeholders
    
**Kanban use case** A DevOps/SRE team handles incoming production issues, bug reports and support requests. Tasks arrived unpredictably.  
  - all incoming tickets are added to Kanban board with columns: "To do ", "In progress", "In Review" , "Done"
  - Each engineer pulls in new work as capacity allows
  - WIP limits set so no ore than say 3 tickets are in "In progress" at once 
10. **What is a "sprint" in Scrum?**
    A sprint is a short, fixed-length period (usually 1-4 weeks) during which a team works to complete a specific amount of work.

11. **What is a "backlog"?**
    A backlog is a prioritized list of all the work that needs to be done on a project. It can contain user stories, features, bug fixes, or other tasks.

12. **What is a "stand-up" meeting?**
    A daily stand-up is a brief, 15-minute meeting where team members share what they did yesterday, what they will do today, and any obstacles they are facing.

13. **What are WIP limits in Kanban?**
    WIP (Work in Progress) limits are a core principle of Kanban that restrict the number of tasks that can be in a particular workflow stage at any time. This helps prevent bottlenecks and focuses the team on finishing work.

---

### **Advanced Concepts**

14. **What is a microservice architecture & "monolith" in software development?**
    Microservices is an architectural style where a large application is broken down into a collection of smaller, independently deployable services. This makes the application easier to develop, test, and scale.
    
    A monolith is a traditional application that is built as a single, unified unit. All components are tightly coupled, which can make it difficult to scale and maintain as the application grows.


15. **What is the difference between Continuous Delivery and Continuous Deployment?**
    With Continuous Delivery, code is ready to be deployed to production at any time, but a human must manually trigger the final deployment. With Continuous Deployment, the code is automatically deployed to production as soon as it passes all tests.

**Use Case: The E-Commerce Store**

An e-commerce company releases new features and bug fixes every two weeks. They use a Continuous Delivery pipeline.

1. A developer pushes code for a new product page.
2. The pipeline automatically builds the code, runs all tests, and deploys it to a staging environment.
3. The Quality Assurance (QA) team tests the feature on the staging environment.
4. A product manager reviews the changes and approves the release.
5. On a scheduled release day, the product manager logs into the CI/CD platform (like Jenkins or GitLab CI) and manually clicks the "Deploy to Production" button.

    
| | **Continuous Delivery** | **Continuous Deployment** |
| :--- | :--- | :--- |
| **Final Step** | Manual trigger | Automatic trigger |
| **Risk Tolerance** | Lower risk, more control | Higher risk, requires trust in automation |
| **Common Use Case** | Scheduled releases, enterprise applications | High-velocity startups, SaaS platforms |
| **Key Benefit** | Controlled, predictable releases | Maximum speed and efficiency |



16. **What are the different types of deployment strategies?**
    Different deployment strategies are used to manage risk and minimize downtime when releasing new software versions. Here are some of the most common ones:

    * **Rolling Deployment:** The new version of the application is gradually deployed to a small number of servers or instances at a time. Once a small batch is updated and verified, the process continues until all instances are running the new version. This minimizes downtime but can complicate rollbacks.

    * **Blue/Green Deployment:** Two identical production environments are maintained. The "Blue" environment runs the current version, and the "Green" environment runs the new version. Once the Green environment is tested and ready, all live traffic is switched from Blue to Green. This allows for zero downtime and very fast rollbacks by simply switching the traffic back.

    * **Canary Deployment:** A new version is released to a very small subset of users (the "canary" group). This group is often a specific geographical region or a small percentage of total users. The new version is monitored closely for errors or performance issues. If all is well, the new version is progressively rolled out to more users. This reduces the risk of a bad release affecting everyone.

    * **Shadow Deployment:** A new version of the application runs in parallel with the live production environment but doesn't serve any traffic to end-users. Instead, a copy of the live production traffic is sent to the new version, allowing the team to test its performance and behavior with real-world load without any risk to the end-user experience.

    * **A/B Testing:** This is a variant of a canary deployment, but with a focus on user experience. Two different versions of a feature (A and B) are released to different user groups to see which one performs better based on specific metrics (e.g., user engagement, click-through rates). The version that wins the test is then rolled out to all users.
    
