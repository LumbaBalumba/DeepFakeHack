#!/bin/bash

while [[ true ]]; do
	sleep 1000
	git add model_weights
	git commit -m "more epochs"
	git push
done
