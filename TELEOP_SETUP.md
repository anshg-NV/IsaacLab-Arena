# Teleop Hardware Setup

These are instructions for setting up a Meta Quest 3 for robot teleop in IsaacLab-Arena.    

## Critical Rules

**READ THE FOLLOWING RULES BEFORE PROCEEDING TO PLUG ANYTHING IN. THESE MISTAKES CAN LOCK YOU OUT OF YOUR MACHINE OR GET YOUR NETWORK PORT DISABLED BY IT!!**

1. **ONLY plug the router into your machine.** Do **NOT** connect it to your desk's wall Ethernet port or to a network switch. IT will auto-disable any port that the router is plugged into.
2. **NEVER reboot your machine with the router plugged in.** When booting up, your machine will default to the router instead of the corporate network and will be unable to decrypt its disk.

## 1. Router

1. Power on the router.
2. Connect the router to your machine via Ethernet (MACHINE ONLY, read critical rules above).
3. Try connecting to the Wi-Fi network listed on the ASUS router sticker. If the connection is successful, you may move onto the Meta Quest setup.
4. Open the admin page at [192.168.50.1](http://192.168.50.1) (for the ASUS router) and log in using the credentials on the sticker on the router.

> My router came pre-configured, so I'm not sure what else needs to be done to set up the router. The admin page can help.

## 2. Meta Quest

1. Power on the Meta Quest and follow the setup instructions.
2. In paralle, on your phone, download the **Meta Horizon** app and create an account with your Nvidia email.
3. When connecting the Meta Quest to Wi-Fi, connect it to `NV_VISITOR`. Do **NOT** connect it to your router!
4. Once the Meta Quest is fully setup and updated, connect it to your router's Wi-Fi network. You can now begin the IsaacLab-Arena teleop setup!
