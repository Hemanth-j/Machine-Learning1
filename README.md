# Machine-Learning1
List of machine Learning projects

cn 
1: Write a NS3 program to connect two nodes with a point to pint link, which have unique 
interface. Analyze the network performance using UDP client server

code:
#include "ns3/core-module.h""\m"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/netanim-module.h"
using namespace ns3;
NS_LOG_COMPONENT_DEFINE ("FirstScriptExample");
int
main (int argc, char *argv[])
{
 CommandLine cmd;
 cmd.Parse (argc, argv);
 Time::SetResolution (Time::NS);
 LogComponentEnable ("UdpEchoClientApplication", LOG_LEVEL_INFO);
 LogComponentEnable ("UdpEchoServerApplication", LOG_LEVEL_INFO);
 NodeContainer nodes;
 nodes.Create (2);
 PointToPointHelper pointToPoint;
 pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
 pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));
 NetDeviceContainer devices;
 devices = pointToPoint.Install (nodes);
 InternetStackHelper stack;
 stack.Install (nodes);
 Ipv4AddressHelper address;
 address.SetBase ("10.1.1.0", "255.255.255.0");
Ipv4InterfaceContainer interfaces = address.Assign (devices);
 UdpEchoServerHelper echoServer (9);
 ApplicationContainer serverApps = echoServer.Install (nodes.Get (1));
 serverApps.Start (Seconds (1.0));
 serverApps.Stop (Seconds (10.0));
 UdpEchoClientHelper echoClient (interfaces.GetAddress (1), 9);
 echoClient.SetAttribute ("MaxPackets", UintegerValue (1));
 echoClient.SetAttribute ("Interval", TimeValue (Seconds (1.0)));
 echoClient.SetAttribute ("PacketSize", UintegerValue (1024));
 ApplicationContainer clientApps = echoClient.Install (nodes.Get (0));
 clientApps.Start (Seconds (2.0));
 clientApps.Stop (Seconds (10.0));
 AnimationInterface anim ("first.xml");
 anim.SetConstantPosition(nodes.Get (0), 10.0, 10.0);
 anim.SetConstantPosition(nodes.Get (1), 20.0, 30.0);
 Simulator::Run ();
 Simulator::Destroy ();
 return 0;
}

execution: ./waf --run scratch/second

2:Write a NS 3 program to demonstrate bus topology. Analyze the performance using UDP based 
applications.

code:
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/netanim-module.h"
using namespace ns3;
NS_LOG_COMPONENT_DEFINE ("SecondScriptExample");
int 
main (int argc, char *argv[])
{
 bool verbose = true;
 uint32_t nCsma = 3;
 CommandLine cmd;
 cmd.AddValue ("nCsma", "Number of \"extra\" CSMA nodes/devices", nCsma);
 cmd.AddValue ("verbose", "Tell echo applications to log if true", verbose);
 cmd.Parse (argc,argv);
 if (verbose)
 {
 LogComponentEnable ("UdpEchoClientApplication", LOG_LEVEL_INFO);
 LogComponentEnable ("UdpEchoServerApplication", LOG_LEVEL_INFO);
 }
 nCsma = nCsma == 0 ? 1 : nCsma;
 NodeContainer p2pNodes;
 p2pNodes.Create (2);
 NodeContainer csmaNodes;
 csmaNodes.Add (p2pNodes.Get (1));
 csmaNodes.Create (nCsma);
 PointToPointHelper pointToPoint;
 pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
 pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));
 NetDeviceContainer p2pDevices;
p2pDevices = pointToPoint.Install (p2pNodes);
 CsmaHelper csma;
 csma.SetChannelAttribute ("DataRate", StringValue ("100Mbps"));
 csma.SetChannelAttribute ("Delay", TimeValue (NanoSeconds (6560)));
 NetDeviceContainer csmaDevices;
 csmaDevices = csma.Install (csmaNodes);
 InternetStackHelper stack;
 stack.Install (p2pNodes.Get (0));
 stack.Install (csmaNodes);
 Ipv4AddressHelper address;
 address.SetBase ("10.1.1.0", "255.255.255.0");
 Ipv4InterfaceContainer p2pInterfaces;
 p2pInterfaces = address.Assign (p2pDevices);
 address.SetBase ("10.1.2.0", "255.255.255.0");
 Ipv4InterfaceContainer csmaInterfaces;
 csmaInterfaces = address.Assign (csmaDevices);
 UdpEchoServerHelper echoServer (9);
 ApplicationContainer serverApps = echoServer.Install (csmaNodes.Get (nCsma));
 serverApps.Start (Seconds (1.0));
 serverApps.Stop (Seconds (10.0));
 UdpEchoClientHelper echoClient (csmaInterfaces.GetAddress (nCsma), 9);
 echoClient.SetAttribute ("MaxPackets", UintegerValue (3));
 echoClient.SetAttribute ("Interval", TimeValue (Seconds (1.0)));
 echoClient.SetAttribute ("PacketSize", UintegerValue (1024));
 ApplicationContainer clientApps = echoClient.Install (p2pNodes.Get (0));
 clientApps.Start (Seconds (2.0));
 clientApps.Stop (Seconds (10.0));
 Ipv4GlobalRoutingHelper::PopulateRoutingTables ();
 pointToPoint.EnablePcapAll ("p2p");
 csma.EnablePcap ("csma1", csmaDevices.Get (1), true);
 csma.EnablePcap ("csma2", csmaDevices.Get (2), true);
 csma.EnablePcap ("csma3", csmaDevices.Get (3), true);
 AnimationInterface anim("bus.xml");
 anim.SetConstantPosition(p2pNodes.Get(0),10.0,10.0);
 anim.SetConstantPosition(csmaNodes.Get(0),20.0,20.0);
 anim.SetConstantPosition(csmaNodes.Get(1),30.0,30.0);
 anim.SetConstantPosition(csmaNodes.Get(2),40.0,40.0);
 anim.SetConstantPosition(csmaNodes.Get(3),50.0,50.0);
 Simulator::Run ();
 Simulator::Destroy ();
 return 0;
}

execution: same as b4

3:Write a NS 3 program to demonstrate star topology. Analyze the performance using UDP based 
applications.

code:
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/netanim-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-layout-module.h"
using namespace ns3;
NS_LOG_COMPONENT_DEFINE ("Star123");
int 
main (int argc, char *argv[])
{
 //
 // Set up some default values for the simulation.
 //
 Config::SetDefault ("ns3::OnOffApplication::PacketSize", UintegerValue (137));
 // ??? try and stick 15kb/s into the data rate
 Config::SetDefault ("ns3::OnOffApplication::DataRate", StringValue ("14kb/s"));
 //
 // Default number of nodes in the star. Overridable by command line argument.
 //
 uint32_t nSpokes = 8;
 std::string animFile1 = "star-animation1.xml";
 CommandLine cmd;
 cmd.AddValue ("nSpokes", "Number of nodes to place in the star", nSpokes);
cmd.AddValue ("animFile1", "File Name for Animation Output", animFile1);
 cmd.Parse (argc, argv);
 NS_LOG_INFO ("Build star topology.");
 PointToPointHelper pointToPoint;
 pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
 pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));
 PointToPointStarHelper star (nSpokes, pointToPoint);
 NS_LOG_INFO ("Install internet stack on all nodes.");
 InternetStackHelper internet;
 star.InstallStack (internet);
 NS_LOG_INFO ("Assign IP Addresses.");
 star.AssignIpv4Addresses (Ipv4AddressHelper ("10.1.1.0", "255.255.255.0"));
 NS_LOG_INFO ("Create applications.");
uint16_t port = 50000;
 Address hubLocalAddress (InetSocketAddress (Ipv4Address::GetAny (), port));
 PacketSinkHelper packetSinkHelper ("ns3::TcpSocketFactory", hubLocalAddress);
 ApplicationContainer hubApp = packetSinkHelper.Install (star.GetHub ());
 hubApp.Start (Seconds (1.0));
 hubApp.Stop (Seconds (10.0));
 //
 // Create OnOff applications to send TCP to the hub, one on each spoke node.
 //
 OnOffHelper onOffHelper ("ns3::TcpSocketFactory", Address ());
 onOffHelper.SetAttribute ("OnTime", StringValue 
("ns3::ConstantRandomVariable[Constant=1]"));
 onOffHelper.SetAttribute ("OffTime", StringValue 
("ns3::ConstantRandomVariable[Constant=0]"));
 ApplicationContainer spokeApps;
 for (uint32_t i = 0; i < star.SpokeCount (); ++i)
 {
 AddressValue remoteAddress (InetSocketAddress (star.GetHubIpv4Address (i), port));
 onOffHelper.SetAttribute ("Remote", remoteAddress);
 spokeApps.Add (onOffHelper.Install (star.GetSpokeNode (i)));
 }
 spokeApps.Start (Seconds (1.0));
 spokeApps.Stop (Seconds (10.0));
 NS_LOG_INFO ("Enable static global routing.");
 //
 // Turn on global static routing so we can actually be routed across the star.
 //
 Ipv4GlobalRoutingHelper::PopulateRoutingTables ();
 NS_LOG_INFO ("Enable pcap tracing.");
 //
 // Do pcap tracing on all point-to-point devices on all nodes.
 //
 pointToPoint.EnablePcapAll ("star123");
// Set the bounding box for animation
 star.BoundingBox (1, 1, 100, 100);
 // Create the animation object and configure for specified output
 AnimationInterface anim (animFile1);
 NS_LOG_INFO ("Run Simulation.");
Simulator::Run ();
 Simulator::Destroy ();
 NS_LOG_INFO ("Done.");
 return 0;
}

execution: same as b4

4:Write a NS3 program to implement FTP using TCP bulk transfer, Analyze the performance 
code:
#include <string>
#include <fstream>
#include "ns3/core-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/network-module.h"
#include "ns3/packet-sink.h"
using namespace ns3;
NS_LOG_COMPONENT_DEFINE ("TcpBulkSendExample");
int
main (int argc, char *argv[])
{
 bool tracing = false;
 uint32_t maxBytes = 0;
// Allow the user to override any of the defaults at
// run-time, via command-line arguments
 CommandLine cmd;
 cmd.AddValue ("tracing", "Flag to enable/disable tracing", tracing);
 cmd.AddValue ("maxBytes",
 "Total number of bytes for application to send", maxBytes);
 cmd.Parse (argc, argv);
// Explicitly create the nodes required by the topology (shown above).
 NS_LOG_INFO ("Create nodes.");
 NodeContainer nodes;
 nodes.Create (2);
 NS_LOG_INFO ("Create channels.");
// Explicitly create the point-to-point link required by the topology (shown above).
 PointToPointHelper pointToPoint;
 pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("500Kbps"));
 pointToPoint.SetChannelAttribute ("Delay", StringValue ("5ms"));
 NetDeviceContainer devices;
 devices = pointToPoint.Install (nodes);
// Install the internet stack on the nodes
 InternetStackHelper internet;
 internet.Install (nodes);
// We've got the "hardware" in place. Now we need to add IP addresses.
 NS_LOG_INFO ("Assign IP Addresses.");
 Ipv4AddressHelper ipv4;
 ipv4.SetBase ("10.1.1.0", "255.255.255.0");
Ipv4InterfaceContainer i = ipv4.Assign (devices);
 NS_LOG_INFO ("Create Applications.");
// Create a BulkSendApplication and install it on node 0
 uint16_t port = 9; // well-known echo port number
 BulkSendHelper source ("ns3::TcpSocketFactory",
 InetSocketAddress (i.GetAddress (1), port));
 // Set the amount of data to send in bytes. Zero is unlimited.
 source.SetAttribute ("MaxBytes", UintegerValue (maxBytes));
 ApplicationContainer sourceApps = source.Install (nodes.Get (0));
 sourceApps.Start (Seconds (0.0));
 sourceApps.Stop (Seconds (10.0));
// Create a PacketSinkApplication and install it on node 1
 PacketSinkHelper sink ("ns3::TcpSocketFactory",
 InetSocketAddress (Ipv4Address::GetAny (), port));
 ApplicationContainer sinkApps = sink.Install (nodes.Get (1));
 sinkApps.Start (Seconds (0.0));
 sinkApps.Stop (Seconds (10.0));
// Set up tracing if enabled
 if (tracing)
 {
 AsciiTraceHelper ascii;
 pointToPoint.EnableAsciiAll (ascii.CreateFileStream ("tcp-bulk-send.tr"));
 pointToPoint.EnablePcapAll ("tcp-bulk-send", false);
 }
// Now, do the actual simulation.
 NS_LOG_INFO ("Run Simulation.");
 Simulator::Stop (Seconds (10.0));
 Simulator::Run ();
 Simulator::Destroy ();
 NS_LOG_INFO ("Done.");
 Ptr<PacketSink> sink1 = DynamicCast<PacketSink> (sinkApps.Get (0));
 std::cout << "Total Bytes Received: " << sink1->GetTotalRx () << std::endl;

execution : same as b4

5:Write a NS3 program to connect two nodes with a point to point link, which have unique 
interface. Analyse the traffic control using TCP by changing suitable parameters.
code:
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/netanim-module.h"
//step1: add the following header files
#include "ns3/flow-monitor.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/traffic-control-module.h"
using namespace ns3;
NS_LOG_COMPONENT_DEFINE ("FirstScriptExample");
int
main (int argc, char *argv[])
{
//step2: declare the variable tracing
bool tracing = false;
CommandLine cmd;
 cmd.Parse (argc, argv);
 
 Time::SetResolution (Time::NS);
 LogComponentEnable ("UdpEchoClientApplication", LOG_LEVEL_INFO);
 LogComponentEnable ("UdpEchoServerApplication", LOG_LEVEL_INFO);
 NodeContainer nodes;
 nodes.Create (2);
 PointToPointHelper pointToPoint;
 pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
 pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));
 NetDeviceContainer devices;
 devices = pointToPoint.Install (nodes);
 InternetStackHelper stack;
 stack.Install (nodes);
 Ipv4AddressHelper address;
 address.SetBase ("10.1.1.0", "255.255.255.0");
 Ipv4InterfaceContainer interfaces = address.Assign (devices);
 UdpEchoServerHelper echoServer (9);
 ApplicationContainer serverApps = echoServer.Install (nodes.Get (1));
 serverApps.Start (Seconds (1.0));
 serverApps.Stop (Seconds (10.0));
 UdpEchoClientHelper echoClient (interfaces.GetAddress (1), 9);
 echoClient.SetAttribute ("MaxPackets", UintegerValue (6));
 echoClient.SetAttribute ("Interval", TimeValue (Seconds (1.0)));
 echoClient.SetAttribute ("PacketSize", UintegerValue (1024));
 ApplicationContainer clientApps = echoClient.Install (nodes.Get (0));
 clientApps.Start (Seconds (2.0));
 clientApps.Stop (Seconds (10.0));
//step3: add the following code for Flow monitor
Ptr<FlowMonitor> flowMonitor;
FlowMonitorHelper flowHelper;
flowMonitor = flowHelper.InstallAll();
Simulator::Stop(Seconds(10.0));
if (tracing==true)
 {
 
pointToPoint.EnablePcapAll ("p2p");
}
Simulator::Run ();
//step 4: add the following statement for xml file
flowMonitor->SerializeToXmlFile("newprg6.xml", true, true);
 Simulator::Destroy ();
 return 0;
}

execution: ./waf --run scratch/newprg6
           python flowmon-parse-results.py newprg6.xml
           
