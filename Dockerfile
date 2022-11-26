FROM spacecloud.unibap.com/unibap/framework-missionimage

SHELL ["/bin/bash", "-c"]

## Do some updates 
# RUN apt-get update -qq \
# 	&& apt-get install -qq -y --no-install-recommends \
# 	libgdal20 python3-rasterio && \
# 	rm -rf /var/lib/apt/lists/*
# WORKDIR /app
# RUN python3.7 -m pip install geopandas

# pytorch lightning needs tensorboard
RUN python3.7 -m pip install tensorboard==2.11.0
RUN python3.7 -m pip install matplotlib==3.5.3

COPY ./ravaen_payload ./ravaen_payload
# COPY ./weights ./weights

