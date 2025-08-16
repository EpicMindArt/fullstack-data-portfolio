# # В requests:
# proxies = {
#    'http': 'http://user:password@proxy.server:port',
#    'https': 'https://user:password@proxy.server:port',
# }
# requests.get('http://example.org', proxies=proxies)

# # В playwright:
# browser = p.chromium.launch(
#     proxy={
#         "server": "http://proxy.server:port",
#         "username": "user",
#         "password": "password"
#     }
# )