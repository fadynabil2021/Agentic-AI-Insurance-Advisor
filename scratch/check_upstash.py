import upstash_redis
print(dir(upstash_redis))
try:
    from upstash_redis import Redis
    print("Found Redis in upstash_redis")
except ImportError:
    print("Redis NOT found in upstash_redis")

try:
    from upstash_redis import AsyncRedis
    print("Found AsyncRedis in upstash_redis")
except ImportError:
    print("AsyncRedis NOT found in upstash_redis")

try:
    import upstash_redis.asyncio
    print("Found upstash_redis.asyncio")
    print(dir(upstash_redis.asyncio))
except ImportError:
    print("upstash_redis.asyncio NOT found")
