# Practical Assignment-group_2

## Setup and Configuration
### Running with Docker(with image: icarusgao/wdps)
This image was built based on karmaresearch/wdps2, you can test the code with this image to save time in install packages.
#### 1. Pull image
   ```
   docker pull icarusgao/wdps
   ```

#### 2. Clone the repository
   ```
   git clone https://github.com/GordonGao2001/group_2.git
   cd group_2
   ```
#### 3. Run the Container
   ```
   docker run -it -v /path/to/local/directory/group_2:/home/user/workspace icarusgao/wdps
   
   ```
#### 4. Run the code
The code is located in workspace, in container terminal:
   ```
   cd workspace
   python main.py
   ``` 
#### 5. Check the result
Output file is in the same directory
   ```
   cat output.txt
   ```


### Running with Docker(with image: karmaresearch/wdps2)
Based on the course-provided image, follow the complete operation process. Note that some package versions may not match your operating system and may require manual handling.
#### 1. Pull image
   ```
   docker pull karmaresearch/wdps2
   ```
#### 2. Clone the repository
   ```
   git clone https://github.com/GordonGao2001/group_2.git
   cd group_2
   ```
#### 3. Run the Container
   ```
   docker run -it -v /path/to/local/directory/group_2:/home/user/workspace karmaresearch/wdps2
   ```
### 4. Prepare 
   ```
   cd workspace
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

#### 4. Run the code
The code is located in workspace, in container terminal:
   ```
   python main.py
   ``` 
#### 5. Check the result
Output file is in the same directory
   ```
   cat output.txt
   ```


### Running Locally
Follow these steps to run the project on your local machine:
#### 1. Clone the repository
      ```
      git clone https://github.com/GordonGao2001/group_2.git
      cd group_2
      ```
#### 2. Create virtual environment with Anoconda
    ```
    conda create --name wdps -c conda-forge python=3.11.2
    conda activate wdps
    ```
#### 3. Environment Setup(Debian/Ubuntu)
    ```
    chmod 777 set_up.sh
    ./set_up.sh
    ```
#### 4. Run the code
The code is located in workspace, in container terminal:
   ```
   python main.py
   ``` 
#### 5. Check the result
Output file is in the same directory
   ```
   cat output.txt
   ```