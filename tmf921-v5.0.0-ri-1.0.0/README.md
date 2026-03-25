# TM Forum Open API – Virtual Service (VS)

This is a **Reference Implemenation (RI)** for simulating a TM Forum Open API and understanding the API. 

It is based on the OpenAPI specification and can be used to **mock the behavior** of the API when a real implementation is not yet available.

This is ideal for:
- Testing client applications early in development
- Frontend/backend decoupling
- Contract-based integration validation
- Viewing a sunny-day result

---

## Prerequisites

- **Docker Desktop** is installed and running.
- On **Windows 11**, ensure:
  - Virtualization is enabled in BIOS
  - WSL 2 is installed and set as default

Make sure the folder includes:
-  'docker-compose.yaml'
-  `run.sh` or `run.bat` (launcher scripts)

---

## Start the Virtual Service

### Linux/macOS:
```bash
chmod +x run.sh
./run.sh
```

### Windows:
```cmd
run.bat
```

Once started, the service will simulate the API and respond to incoming requests.

---

### View and test the API

You can browse the API documentation and test available endpoints. 

```
Open: `http://localhost:8[api-number]/tmf-api/partyRoleManagement/v5/api-docs/`
```


You can now make HTTP requests to this address using:
- Postman
- curl
- Any API client or application under development

The RI will generate **mock responses** based on the API contract and conformance profile, ensuring that the behavior is predictable and spec-compliant.

---


## Questions?

Contact: [openapi@tmforum.org](mailto:openapi@tmforum.org)
