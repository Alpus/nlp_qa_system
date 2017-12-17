# Question-Answer system

Run jupyter notebook inside the docker container with `run.ipynb` file for testing.

##### Run example:
```bash
docker build -t image_name .
docker run -it -p 8888:8888 image_name

# or

docker run -it -p 8888:8888 alpus/question_answer_system
```

Jupyter notebook uses `8888` port inside docker.
