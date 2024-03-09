#!/bin/bash

if [ "$EUID" -ne 0 ]; then
    echo "This script must be run with sudo."
    echo "Use 'sudo ./wifi_autorun.sh' instead of './wifi_autorun.sh'"
    echo "Exiting..."
    exit 1
fi

# You can add other cron jobs to crontab(root)

# Define the first cron job and its schedule
cron_job1="@reboot /bin/bash /home/$(logname)/ugv_pt_rpi/wifi_ctrl.sh --auto >> /home/$(logname)/wifi.log 2>&1"

# Check if the first cron job already exists in the user's crontab
if crontab -l | grep -q "$cron_job1"; then
    echo "First cron job is already set, no changes made."
else
    # Add the first cron job for the user
    (crontab -l 2>/dev/null; echo "$cron_job1") | crontab -
    echo "First cron job added successfully."
fi

echo "Change default wifi mode: sudo crontab -e"
echo "--ap --auto(default) --sta"
echo "Change ugv_pt_rpi/wifi_ctrl.sh to edit wifi config."