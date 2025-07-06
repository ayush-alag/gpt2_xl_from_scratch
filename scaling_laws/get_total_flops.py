import requests

def get_total_flops(config):
    response = requests.get("http://hyperturing.stanford.edu:8000/total_flops_used", config)
    return response.json()

if __name__ == "__main__":
    DEFAULT_API_KEY = """ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDCj8Lkwc96MtE2JjXLIgF1mLCI97JpoqeUKx9mszDt0xYuQ8HOGqa2D5SgBSeha+dG182xGnAAJ961989Ex8D56FkqMifGhTJSYrJh50G8RV+4MQJyRg9kiOvB7Vdbe/gOmKC1cZDLbJhvbgdmhiJcHslq++PbfNP8a21dcQd35kbAKMaU0kGsI5khuScMQ0VbEapAjxIriwmNVZFgQjixvlqUvawl3Nm4UQisjkPt3M+OobLNcp0Zm/4fjategtlJ0KwfG8EnadBjMIriVW3JNCurB9qH+EkOqKXEj2WM4++QwOmFyiWYJfD26olwj9DJSu534hGZQFrI2nI8+js8xLzqHCL0tXesnB1bSIYIyoDN6uTmmsN/ZpdxPyiv+oBnBXOlJRcb8hYRG5i9DWL0dq2HD+NsN26M94c8vxd+iuORas5SS5FFSLXMx1hj8a7GCsDo2iiw4NBZ68HENZMJ4jUy63cevqbkjklJKPLGVeaJTg0WZM/fJj/XEUW4C8s= ayushalag@Ayushs-MacBook-Pro.local"""

    api_key_config = {
        "api_key": DEFAULT_API_KEY
    }

    total_flops = get_total_flops(api_key_config)
    print(total_flops)
    print(f"Total FLOPs: {total_flops}")
    print("FLOP Percentage: ", total_flops / 2e18)