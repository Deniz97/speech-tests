deps:
	pip install -r requirements.txt

dev:
	python main.py

create-env:
	python3.10 -m venv .venv

ip=58.79.62.172
ssh:
	ssh -p 61705 -i ./id_rsa root@$(ip) -L 8080:localhost:8080
# ssh -p 61705 -i ./id_rsa root@58.79.62.172 -L 8080:localhost:8080
	

scp:
	scp -P 61705 -i ./id_rsa ./sesler2.zip root@$(ip):/root/sesler2.zip
# scp -P 61705 -i ./id_rsa ./sesler2.zip root@58.79.62.172:/root/sesler2.zip

source:
	source .venv/bin/activate

unzip:
	apt update
	apt install -y unzip
	unzip sesler2.zip

clone:
	git clone https://github.com/Deniz97/speech-tests.git

install_python_310:
	apt update
	apt install -y software-properties-common
	add-apt-repository ppa:deadsnakes/ppa
	apt update
	sudo apt install python3.10 python3.10-venv python3.10-dev
	
