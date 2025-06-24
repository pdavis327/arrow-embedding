#!/bin/bash

# Create htpasswd file for demo users
htpasswd -c -B -b users.htpasswd demo-user demo123
htpasswd -B -b users.htpasswd admin-user admin123
htpasswd -B -b users.htpasswd arrow-user arrow123

echo "Created htpasswd file with demo users"
