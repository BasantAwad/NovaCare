"""Centralized robot configuration values.

Place global robot settings here. The value is read from the `ROBOT_IP`
environment variable if set, otherwise falls back to the default below.
"""
import os

# Change the default here to update all scripts that import this value.
ROBOT_IP = os.getenv('ROBOT_IP', '192.168.8.50')
