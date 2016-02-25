#!/bin/bash

# remove leading white spaces, trade whitespace for column separator |, add | to end of line
sed -e 's/^[[:space:]]\+//g' -e 's/[[:space:]][[:space:]]*/|/g' -e 's/$/|/g'