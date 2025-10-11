## A Beginner's Guide to Networking 🌐
> **Author: Sudish**
> 
## What is a Computer Network? 

At its simplest, a **computer network** is a group of interconnected devices that can communicate with one another to share resources and information. Think of it like a public transportation system for data. The devices (like your computer, phone, or a printer) are the stations, and the cables or wireless signals are the routes. The goal is to move data from one point to another efficiently and securely.

### Key Components of a Network

To get our "data transportation system" running, we need a few essential building blocks:

  * **Nodes**: Any device connected to the network is a node. This can be a laptop, a printer, a server, or even a switch or router.
  * **Links**: These are the connections between nodes. They can be physical (like an **Ethernet cable** or a fiber optic cable) or wireless (like Wi-Fi).
  * **Protocols**: These are the rules that govern how devices communicate. They are the "language" of the network. Without them, devices wouldn't understand each other.

-----

## What is a Switch?

In networking, a **switch** is a hardware device that connects multiple devices within a network, enabling them to communicate with each other. It acts as a central hub, receiving data packets and forwarding them to the correct destination based on their MAC addresses, ensuring efficient data transfer and managing network traffic.

**Function:**
Switches are essential for connecting devices like computers, printers, servers, and other network-enabled devices. They allow these devices to share resources and communicate with each other.

**Packet Switching:**
Switches use **packet switching**, a method where data is broken down into small packets, each containing the destination address, and then transmitted across the network.

**MAC Address:**
Switches learn the **MAC addresses** of the connected devices and store them in a MAC address table. This table helps the switch determine the appropriate port to forward data packets to their intended recipients.

**Data Transfer:**
When a data packet arrives at a switch port, the switch examines the destination MAC address and forwards the packet only to the port where the intended recipient is connected. This efficient forwarding prevents unnecessary traffic and congestion.

**Network Segmentation:**
Switches can also be used to create separate network segments or subnets, allowing for improved security and performance.

**Comparison with Hubs:**
Unlike hubs, which transmit data to all connected devices regardless of the destination, switches only forward data to the intended recipient, resulting in more efficient and less congested networks.

-----

## Types of Network Switches

**1. Unmanaged Switch**
Definition: A plug-and-play device with no configuration options.
Use Case: Ideal for home networks or small businesses requiring basic connectivity.
Features: Automatically handles data traffic without user intervention.

**2. Managed Switch**
Definition: A switch that offers advanced features for network management.
Use Case: Suitable for enterprise networks needing control over traffic.
Features: Supports VLANs, SNMP monitoring, QoS, and port mirroring.

**3. Smart Switch (Lightly Managed Switch)**
Definition: A switch that provides limited management features.
Use Case: Ideal for small to medium-sized businesses needing some control without complexity.
Features: Supports basic configurations like VLANs and QoS.

**4. PoE Switch (Power over Ethernet)**
Definition: A switch that delivers electrical power over Ethernet cables.
Use Case: Useful for powering devices like IP cameras, VoIP phones, and wireless access points.
Features: Eliminates the need for separate power supplies for connected devices.

-----

## Layer 2 and Layer 3 Switches

**Layer 2 Switch**
Definition: Operates at the Data Link Layer, forwarding data based on MAC addresses.
Use Case: Suitable for segmenting networks into separate collision domains.
Features: Provides efficient data forwarding within the same network.

**Layer 3 Switch**
Definition: Combines switching and routing functionalities, operating at the Network Layer.
Use Case: Ideal for inter-VLAN routing and managing large, complex networks.
Features: Supports routing protocols and IP address-based forwarding.

-----

## How to Configure a Switch

Start by connecting to its console or establishing a remote connection like SSH. Next, you'll set up basic configurations like hostname, IP address, and default gateway. Then, you can configure VLANs, trunk ports, and other advanced features as needed. Finally, you'll save the configuration to ensure it's persistent.

1.  **Establish a Connection:**

      * **Console:** Connect a cable to the console port on the switch and connect the other end to your PC. Use a console cable and terminal emulator (e.g., PuTTY) to connect.
      * **SSH:** If SSH is enabled, you can connect remotely using a terminal program.
      * **Telnet:** While Telnet is less secure, it can be used for initial access if SSH is not yet enabled.

2.  **Basic Configuration:**

      * **Hostname:** Assign a unique name to the switch for easier identification.
      * **IP Address:** Set a static IP address, subnet mask, and default gateway to enable remote management and communication.
      * **Password Security:** Set passwords for console, VTY lines, and privileged EXEC mode.

3.  **Advanced Features:**

      * **VLANs:** Create virtual LANs to segment your network traffic.
      * **Trunk Ports:** Configure ports to carry traffic for multiple VLANs.
      * **Access Control Lists (ACLs):** Control traffic flow based on source/destination IP addresses or other criteria.
      * **Quality of Service (QoS):** Prioritize different types of network traffic.

4.  **Save the Configuration:**
    Use the command `copy running-config startup-config` to save the current configuration to the switch's flash memory.

-----

## VLAN

A **VLAN** is a logical grouping of devices within a Layer 2 network (such as a switch), where each group behaves as if it were on its own separate physical network. VLANs improve security, performance, and manageability.

**Why use VLANs?**

  * **Segmentation:** Separate departments (HR, Finance, IT) even on the same switch.
  * **Security:** Devices in different VLANs can’t communicate unless routed.
  * **Reduced Broadcast Domains:** Reduces unnecessary traffic.
  * **Efficiency:** Better bandwidth utilization.

### VLAN Configuration Commands (Cisco IOS)

```
➤ 1. Create VLAN
Switch(config)# vlan 10
Switch(config-vlan)# name HR
Switch(config-vlan)# exit

➤ 2. Assign VLAN to Interface (Access Port)
Switch(config)# interface FastEthernet 0/1
Switch(config-if)# switchport mode access
Switch(config-if)# switchport access vlan 10

➤ 3. Configure a Trunk Port
Switch(config)# interface GigabitEthernet 0/24
Switch(config-if)# switchport mode trunk
Switch(config-if)# switchport trunk native vlan 1
Switch(config-if)# switchport trunk allowed vlan 10,20,30

➤ 4. Verify VLANs
Switch# show vlan brief
Switch# show interfaces trunk
```

### Enabling SSH for Secure Management

```
Set Domain Name:
    Switch(config)# ip domain-name [DOMAIN_NAME]
Generate RSA Keys:
    Switch(config)# crypto key generate rsa
Configure VTY Lines for SSH:
    Switch(config)# line vty 0 4
    Switch(config-line)# transport input ssh
    Switch(config-line)# login local
Create User Credentials:
    Switch(config)# username [USERNAME] privilege 15 secret [PASSWORD]
```

### Monitoring and Verification Commands

```
View Current Configuration:
    Switch# show running-config
Check VLAN Information:
    Switch# show vlan brief
Display Interface Status:
    Switch# show interfaces status
Verify Trunk Ports:
    Switch# show interfaces trunk
Check STP Status:
    Switch# show spanning-tree
```



### How a Switch Works

A switch operates by using **packet switching**, where data is broken into small chunks called **packets**, each with a destination address. The key to a switch's efficiency is the **MAC (Media Access Control) address**, a unique physical address assigned to every network-enabled device. The switch learns the MAC addresses of all connected devices and stores them in a **MAC address table**.

When a packet arrives at a switch, the switch checks the packet's destination MAC address. It then looks up that address in its table to find the corresponding port. The packet is then forwarded only to that specific port. This **efficient data transfer** is what makes switches so much better than older devices like hubs, which simply broadcast all data to all connected devices, causing unnecessary traffic and slowdowns.

Switches also allow for **network segmentation**, which means you can create separate network segments or subnets for better security and performance. For example, you can create a separate segment for your guest Wi-Fi to keep it isolated from your main network.

-----
**Switch & Cables pics :**

<img width="700" height="334" alt="image" src="https://github.com/user-attachments/assets/e6bec846-f72b-4e56-aca5-92132e0deb61" />
<img width="700" height="318" alt="image" src="https://github.com/user-attachments/assets/9cadd89b-797a-4873-b2a6-2619c08c68b9" />

<img width="700" height="386" alt="image" src="https://github.com/user-attachments/assets/de203fc2-6cda-49d0-830f-04ff86bc6859" />


-----

## Understanding VLANs

A **VLAN** (Virtual Local Area Network) is a logical grouping of devices that behaves as if it's on its own separate network, even if all the devices are connected to the same physical switch. VLANs are incredibly useful for improving network security, performance, and manageability.

<img width="700" height="700" alt="image" src="https://github.com/user-attachments/assets/c5d0a0a3-0585-470c-855d-1d3f7d50c404" />

### Why Use VLANs?

  * **Segmentation**: VLANs allow you to separate different departments (e.g., HR, Finance, IT) within an organization. This keeps their traffic separate and secure.
  * **Security**: Devices in different VLANs cannot communicate with each other unless a Layer 3 device (like a Layer 3 switch or router) is configured to route traffic between them.
  * **Reduced Broadcasts**: VLANs limit the size of broadcast domains. When a device sends a broadcast, it only reaches devices within the same VLAN, reducing unnecessary network traffic.

### Basic VLAN Configuration Commands (Cisco IOS)

Here are some common commands to get you started with VLANs on a Cisco switch:

1.  **Create a VLAN**:
    ```bash
    Switch(config)# vlan 10
    Switch(config-vlan)# name HR
    ```
2.  **Assign a Port to a VLAN**:
    ```bash
    Switch(config)# interface FastEthernet 0/1
    Switch(config-if)# switchport mode access
    Switch(config-if)# switchport access vlan 10
    ```
3.  **Configure a Trunk Port**:
    ```bash
    Switch(config)# interface GigabitEthernet 0/24
    Switch(config-if)# switchport mode trunk
    Switch(config-if)# switchport trunk allowed vlan 10,20,30
    ```
4.  **Verify VLANs**:
    ```bash
    Switch# show vlan brief
    Switch# show interfaces trunk
    ```
###  Differences Between Access ports and Trunk ports

<img width="700" height="700" alt="image" src="https://github.com/user-attachments/assets/adcbbd95-5b48-4cf8-8acb-2fbaa3f45bc6" />

-----
## FAQ

## What is a Network Switch?

**1. What is the main difference between a switch and a hub?**
A **switch** is a smart device that forwards data only to the intended recipient, while a **hub** broadcasts data to all devices. This makes switches much more efficient and reduces network congestion.

**2. How does a switch know where to send data?**
A switch learns the unique **MAC address** of each device connected to it and stores this information in a MAC address table. When a data packet arrives, it looks up the destination MAC address to find the correct port.

**3. What is a MAC address?**
A **MAC address** (Media Access Control address) is a unique, physical identifier assigned to a network interface card (NIC) by the manufacturer. It’s like a device's permanent street address on a network.

**4. What is the difference between a Layer 2 and a Layer 3 switch?**
A **Layer 2 switch** operates at the Data Link layer and forwards data based on MAC addresses. A **Layer 3 switch** operates at the Network layer and can also route traffic based on IP addresses, much like a router.

**5. Why are Layer 3 switches used?**
Layer 3 switches are used for **inter-VLAN routing** to allow devices in different VLANs to communicate without needing a separate, dedicated router. They are ideal for large, complex networks.

**6. What is a managed switch?**
A **managed switch** offers advanced features and configuration options, allowing network administrators to monitor, control, and prioritize traffic. They are used in enterprise networks.

**7. What is an unmanaged switch?**
An **unmanaged switch** is a simple, plug-and-play device that requires no configuration. It's a "set it and forget it" option for basic home or small office networks.

**8. What is a PoE switch?**
A **PoE (Power over Ethernet) switch** can deliver both data and electrical power to a device over a single Ethernet cable. This is useful for devices like IP cameras and VoIP phones, eliminating the need for a separate power source. 

## Understanding VLANs

**9. What is a VLAN?**
A **VLAN (Virtual Local Area Network)** is a logical group of devices that share the same network segment, even if they're not physically connected to the same switch. It's used to segment a network for security and efficiency.

**10. Why are VLANs important?**
VLANs improve network security by isolating traffic and prevent unnecessary broadcasts from slowing down the network. For example, you can create separate VLANs for an HR department and a finance department to keep their data secure.

**11. What is an access port?**
An **access port** is a switch port that belongs to a single VLAN and can only carry traffic for that specific VLAN. It’s used to connect end devices like computers and printers.

**12. What is a trunk port?**
A **trunk port** is a switch port that can carry traffic for multiple VLANs simultaneously. It's used to connect two switches or a switch and a router.

**13. What is the native VLAN?**
The **native VLAN** is the VLAN that carries untagged traffic on a trunk port. It's the default VLAN for frames that don't have a VLAN ID.

**14. What are some common VLAN configuration commands?**
Common commands include `vlan 10` (to create a VLAN), `switchport mode access` (to set an access port), and `switchport mode trunk` (to set a trunk port).

## Basic Networking Concepts

**15. What is a broadcast domain?**
A **broadcast domain** is a set of devices that can receive broadcast frames from each other. Switches segment broadcast domains, preventing broadcasts from flooding the entire network.

**16. What is the OSI model?**
The **OSI (Open Systems Interconnection) model** is a conceptual framework that standardizes the functions of a telecommunication or computing system into seven layers. It helps people understand how data travels over a network.

**17. What are the most important layers of the OSI model for a switch?**
A **Layer 2 switch** operates primarily at the **Data Link Layer**, while a **Layer 3 switch** operates at both the **Data Link Layer** and the **Network Layer**.

**18. What is the purpose of an IP address?**
An **IP address** (Internet Protocol address) is a logical address that uniquely identifies a device on a network. It's used to route data between different networks and across the internet.

**19. What is the purpose of a default gateway?**
A **default gateway** is the entry and exit point of a network. It is the address of a router or Layer 3 switch that devices use to send traffic to other networks, such as the internet.

**20. What is Spanning Tree Protocol (STP)?**
**STP** is a network protocol that prevents switching loops, which can occur when there are redundant paths between network devices. STP intelligently blocks redundant links to ensure a loop-free topology.


[Networking Basics (2025) | What is a switch, router, gateway, subnet, gateway, firewall & DMZ](https://www.youtube.com/watch?v=_IOZ8_cPgu8)
