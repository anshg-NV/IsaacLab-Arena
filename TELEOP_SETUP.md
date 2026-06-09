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

> My router came pre-configured, so I'm not sure what else needs to be done to set up the router. The admin page mentioned above may help.

## 2. Meta Quest

1. Power on the Meta Quest and follow the setup instructions.
2. In parallel, on your phone, download the **Meta Horizon** app and create an account with your Nvidia email.
3. When connecting the Meta Quest to Wi-Fi, connect it to `NV_VISITOR`. Do **NOT** connect it to your router!
4. Once the Meta Quest is fully setup and updated, connect it to your router's Wi-Fi network. You can now begin the IsaacLab-Arena teleop setup!

## 3. Teleop

For the most part, follow the instructions [here](https://isaac-sim.github.io/IsaacLab-Arena/release/0.2.1/pages/example_workflows/locomanipulation/step_2_teleoperation.html).

Below is a quick TLDR of what you need to do:
1. Verify that local machine and Meta Quest are connected to corporate network and `NV_VISITOR`, respectively.
2. Start CloudXR runtime and demo recording scripts on local machine. Wait for environment to load.
3. Go to https://nvidia.github.io/IsaacTeleop/client in Meta Quest browser.
4. Switch local machine and Meta Quest to router network.
5. In the Isaac Sim window, click on the `XR` tab and start the session.
6. Start collecting demos on the Meta Quest.
7. The teleop dataset will be stored on the Docker container, so make sure to copy it to your machine's storage.
