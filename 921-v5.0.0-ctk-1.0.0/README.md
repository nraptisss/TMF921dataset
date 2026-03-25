# TM Forum Open API – Conformance Test Kit (CTK)

This Conformance Test Kit (CTK) allows you to validate that your Open API implementation conforms to the TM Forum specification for a specific API (e.g., TMF622, TMF679). It tests whether your implementation meets the mandatory and optional features defined in the associated Conformance Profile.

You can use this CTK as part of the **formal certification process** or for internal quality assurance during development.

---

## Prerequisites

- **Docker Desktop** is installed and running.
- On **Windows 11**, make sure that:
  - Virtualization is enabled in the BIOS
  - [WSL 2](https://docs.microsoft.com/en-us/windows/wsl/install) is installed and set as default

---

## Configuration

Before running the CTK, edit the `config.json` file to match your API deployment:

```json
{
    "url": "http://tmf622-ri:8622/tmf-api/productOrderingManagement/v5",
    "headers": {
        "Content-Type": "application/json",
        "Accept": "application/json"
    },
    "payloads": {
        "ProductOrder": {
            "POST": {
                "payload": {
                    "category": "B2C product order",
                    "description": "Product Order illustration sample",
                    "externalId": [
                      {
                        "@type": "ExternalIdentifier",
                        "owner": "TMF",
                        "externalIdentifierType": "POnumber",
                        "id": "456"
                      }
                    ],
                }
            }
        }
    }
}
```

- `url`: Base URL of your API (must be accessible from Docker)
- `headers`: Optional HTTP headers (e.g., authentication)
- `payloads`: Optional request bodies for operations that require them. Use operation names as defined in the OpenAPI spec or conformance profile.

**Tip:** Read the Conformance Profile located in the `conformance/` folder to understand which operations and features will be tested.

---

## Run the CTK Tests

The platform environment will need to be set to run the CTK docker image.
In the majority of cases (inc. windows) it will be:
```
platform=linux/amd64
```

In some cases it might be:
```
platform=linux/arm64
```

### Linux/macOS:
```bash
chmod +x run.sh
platform=linux/amd64 ./run.sh
```

### Windows:
```cmd
set platform=linux/amd64
run.bat
```

You will see test output logs in the terminal. Upon completion, a test report will be generated.

---

## View the Test Results

Open the following file in a browser:

 `REPORT.HTML`

This report shows the test results for each conformance criterion. Review all **mandatory** feature test cases — all must pass for your API to be considered conformant.

---

## Submitting for Certification

How to Request Official TM Forum API Conformance Certification

	1.  Complete the API Certification Report provided by your TM Forum contact.
  2.  Create a ZIP file containing the entire reports folder.
	3.	Send the ZIP file together with the REPORT.HTML file to your TM Forum contact, including the relevant API metadata (e.g., implementation URL, version, platform).
	4.	TM Forum will review the submission and, if all requirements are satisfied, issue an official Conformance Certification Badge.

---

## Troubleshooting

- **Connection errors:** Check that your API endpoint is publicly accessible and that the `url` in `config.json` is correct.
- **Authentication failures:** Ensure valid tokens or credentials are included in `headers`.
- **The CTK is shipped in multiple docker architectures. When running in a windows environment it maybe necessary to modify the platform: declarion ine docker-compose to fit you environmwent.
- for example

platform: linux/amd64
platform: linux/arm64

---

## Questions?

Contact: [openapi@tmforum.org](mailto:openapi@tmforum.org)