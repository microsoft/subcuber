FROM nvcr.io/nvidia/pytorch:26.02-py3
RUN mkdir /strassencc-pldi26-ae/
COPY . /strassencc-pldi26-ae/
RUN pip install tabulate matplotlib