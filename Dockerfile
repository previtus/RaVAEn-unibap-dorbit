FROM spacecloud.unibap.com/unibap/framework-missionimage

SHELL ["/bin/bash", "-c"]

## Do some updates 
# RUN apt-get update -qq \
# 	&& apt-get install -qq -y --no-install-recommends \
# 	libgdal20 python3-rasterio && \
# 	rm -rf /var/lib/apt/lists/*
# RUN python3.7 -m pip install geopandas

# pytorch lightning needs tensorboard
# RUN python3.7 -m pip install tensorboard==2.11.0
# for plotting (likely replace with something simpler...)
# RUN python3.7 -m pip install matplotlib==3.5.3

COPY ./ravaen_payload ./ravaen_payload
COPY ./tile_classifier ./tile_classifier
COPY run_all.sh run_all.sh

#WORKDIR /ravaen_payload
#CMD ["python3.7","run_inference.py"]

WORKDIR /
CMD ["bash","run_all.sh"]

# check image sizes with
#docker image ls
#docker image history ravaen_payload:latest
