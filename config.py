class Config:

    DEBUG = False
    TESTING = False
    SECRET_KEY = "11Fnw8U6DXrMFvbH9jCdZQ"

class ProductionConfig(Config):
    pass

class DevelopmentConfig(Config)
    DEBUG = True