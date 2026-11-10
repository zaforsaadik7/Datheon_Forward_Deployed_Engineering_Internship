# Cloud Infrastructure & VPC Peering Runbook

## 1. Network Topology & Peering
Production workloads run inside the primary VPC (`vpc-prod-compute`) with CIDR block `10.42.0.0/16` in `us-east-1`, peered with the shared services VPC (`vpc-shared-sec`) on `10.10.0.0/16`. All cross-VPC traffic must traverse AWS Transit Gateway `tgw-09a8b7c6d5e4f3210` with strict Network ACL enforcement.

## 2. Routing, NAT Egress & Failover
Outbound internet traffic from private subnets (`10.42.16.0/20` and `10.42.32.0/20`) routes through redundant NAT Gateways deployed across three Availability Zones (`us-east-1a`, `us-east-1b`, `us-east-1c`). BGP dynamic routing over AWS Direct Connect uses ASN `64512` with bidirectional forwarding detection (BFD) configured at 300ms intervals for sub-second failover.
