
## Docker for Absolute Beginners

[Docker Doc](https://docs.docker.com/)

### What is docker?
- virtualized s/w
- makes developing and deploying applications much easier
- packages application with all necessary dependencies , configuration, system tool and runtime

```
Docker is a platform that allows you to build, ship, and run applications in containers.
Containers are standardized, self-sufficient units of software that package up code and all its dependencies so the application runs consistently on any infrastructure.

Key concepts in Docker:

- Docker Image: A read-only template with instructions for creating a container.   
- Docker Container: A running instance of a Docker image.   
- Dockerfile: A text document that contains all the commands a user could call on the command line to assemble an image.   
- Docker Compose: A tool for defining and running multi-container Docker applications
```

### Docker intallation 
- Install docker on your local  machine
  - follow docker offical document for windows or mac
  -   https://docs.docker.com/desktop/setup/install/mac-install/
  -   note : docker desktop includes - Docker engine ( daemon process name "dockerd"). this will used to manage images and container 
  -   Docker CLI "docker"
```
$docker images
$docker ps
$docker pull nginx:1.23   #this will download from Dockerhub
$docker images    # now you can see one image
$docker pull nginx    # this will download latest image
$docker images        # now you can see two images

$ docker run nginx:1.23
$ docker run nginx:1.23      # run in background
# docker ps    - to see running image 
```

### Docker port binding 
- publish container port to the host. Assune your container is running on port 80

```
$docker stop <imageID>
$docker run -d -p 9000:80 nginx:1.23
$docker ps
://https://localhost:90000

$docker run --n name web-app -d -p 9000:80 nginx:1.23

```

### Building your own docker image 
use case - we like to deploy own app using docker container
- dockerfile :Build instruction
Basick structure of Dockerfile
```
# Base Image
FROM <image_name>:<tag>

# Maintainer Information (Optional)
MAINTAINER <your_name> <your_email>

# Commands to Execute
RUN <command>
COPY <src> <dest>
WORKDIR <directory>
EXPOSE <port>
CMD ["command", "arg1", "arg2"]
ENTRYPOINT ["command", "arg1", "arg2"]
```
Breakdown of Commands:

FROM: Specifies the base image to use.
MAINTAINER: Sets the maintainer information.
RUN: Executes a command in the image.
COPY: Copies files or directories from the host to the image.
WORKDIR: Sets the working directory for subsequent commands.
EXPOSE: Exposes a port for the container.
CMD: Sets the default command to run when the container starts.
ENTRYPOINT: Sets the executable file and arguments to run when the container starts.

**Example Dockerfile for a Node.js Application:**

```
# Use a Node.js image as the base
FROM node:18-alpine

# Set the working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# Expose port 3000
EXPOSE 3000


# Start the app
CMD ["node", "index.js"]
```
$ docker build -t my-node-app .
$ docker run -it --rm -p 3000:3000 my-node-app
This command will start a container from the my-node-app image, mapping port 3000 of the container to port 3000 of your host machine.

### Docker FAQ
1. What is a Docker container?
2. How does Docker differ from traditional virtual machines?
3. Explain the concept of Docker images and containers.
4. What is a Dockerfile?
5. How does Docker use layers to optimize image size?
6. What is a Docker registry?
7. Explain the following Docker commands:
```
docker pull
docker push
docker run
docker ps
docker stop
docker rm
docker build
docker images
```
8. How would you create a Docker image for a Node.js application?
9. How would you run a Docker container in detached mode?
10. How would you map a port from a container to the host machine?
11. What is Docker Compose?
12. How do you define services in a Docker Compose file?
13. How do you link services in a Docker Compose file?
14. How do you define networks in a Docker Compose file?
15. What are some security best practices for Docker & How can you secure Docker images?
