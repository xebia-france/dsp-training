# Créer une instance EC2 medium avec Ubuntu 18.04

sudo apt-get update
sudo apt install -y python3-pip

git clone https://github.com/romibuzi/dsp-training.git
cd dsp-training
git checkout exercice6-solution
# To be able to copy/paste files from local to EC2 instance (see copy/paste command at end of file)
sudo chmod 777 -R ../dsp-training

# requirements for airflow
sudo apt-get install -y --no-install-recommends freetds-bin krb5-user ldap-utils libffi6 libsasl2-2 libsasl2-modules \
libssl1.1 locales  lsb-release sasl2-bin sqlite3 unixodbc
pip3 install --upgrade pip==20.2.4

pip3 install -r requirements.txt
AIRFLOW_VERSION=2.0.2
PYTHON_VERSION="$(python3 --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
pip3 install "apache-airflow[async,postgres]==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
export AIRFLOW_HOME=$(pwd)/airflow

export PATH=~/.local/bin/:$PATH

airflow db init

airflow users create \
--username admin \
--firstname Peter \
--lastname Parker \
--role Admin \
--email spiderman@superhero.org

airflow webserver --port 5000

# Dans un autre terminal connecté en ssh
cd dsp-training
export AIRFLOW_HOME=$(pwd)/airflow
export PATH=~/.local/bin/:$PATH
airflow scheduler

# To copy / paste a file from local machine to EC2 instance
# scp -i private_key.pem myfile ubuntu@ec2-<ip-adress>.eu-west-1.compute.amazonaws.com:/home/ubuntu/dsp-training/path-to-folder