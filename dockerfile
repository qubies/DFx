FROM qubies/general
USER root
RUN apt-get update && apt-get install -y python3 python3-pip default-jdk
RUN RUN curl https://sh.rustup.rs -sSf | bash -s -- -y && source $HOME/.cargo/env && rustup default nightly
RUN pip3 install tensorflow tensorflow-hub torch tqdm transformers language_tool_python spacy
RUN python -m spacy download en_core_web_sm
