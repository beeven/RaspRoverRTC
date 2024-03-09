#!/bin/bash

if [ "$EUID" -ne 0 ]; then
    echo "This script must be run with sudo."
    echo "Use 'sudo ./wifi_ctrl.sh' instead of './wifi_ctrl.sh'"
    echo "Exiting..."
    exit 1
fi

STA_SSID="HJSLH-5G"
#STA_SSID="JSBZY-5G"
STA_PASSWORD="waveshare0755"

AP_SSID="RaspRover"     # AP_SSID
AP_PASSWORD="12345678"  # AP_PASSWORD

CONFIG_FILE="/etc/NetworkManager/NetworkManager.conf"

sudo iw dev wlan0 set power_save off

function fix_network_manager {
    if grep -q "unmanaged-devices=interface-name:wlan0" "$CONFIG_FILE"; then
        echo "Fixing NetworkManager configuration..."
        # delete unmanaged-devices
        sudo sed -i '/unmanaged-devices=interface-name:wlan0/d' "$CONFIG_FILE"
        # restart NetworkManager
        sudo systemctl restart NetworkManager
        echo "NetworkManager configuration fixed."
    else
        echo "NetworkManager configuration is OK."
    fi
}

function start_ap {
    echo "disconnect wlan0..."
    sudo nmcli device disconnect wlan0
    echo "Starting Access Point..."
    sudo create_ap wlan0 eth0 "$AP_SSID" "$AP_PASSWORD"
}

function connect_wifi {
    fix_network_manager

    # echo "Connecting to Wi-Fi using wpa_supplicant..."
    # sudo wpa_cli -i wlan0 reconfigure
    echo "disconnect wlan0..."
    sudo nmcli device disconnect wlan0
    echo "Connecting to Wi-Fi..."
    sudo nmcli device wifi connect "$STA_SSID" password "$STA_PASSWORD" ifname wlan0

    # check
    for i in {1..10}; do
        if iw wlan0 link | grep -q 'Connected'; then
            echo "Connected to Wi-Fi."
            return
        fi
        sleep 2
    done

    echo "Failed to connect to Wi-Fi."
}

function auto_mode {
    # check Wi-Fi
    if nmcli device status | grep wlan0 | grep -q "disconnected"; then
        echo "Not connected to Wi-Fi. Trying to connect..."
        connect_wifi

        # check Wi-Fi again
        if nmcli device status | grep wlan0 | grep -q "disconnected"; then
            echo "Failed to connect to Wi-Fi. Starting Access Point..."
            start_ap
        else
            echo "Connected to Wi-Fi."
        fi
    else
        echo "Already connected to a Wi-Fi network."
    fi
}

# use the args
case "$1" in
    --auto)
        auto_mode
        ;;
    --ap)
        start_ap
        ;;
    --sta)
        connect_wifi
        ;;
    *)
        echo "Usage: $0 --auto|--ap|--sta"
        exit 1
        ;;
esac
