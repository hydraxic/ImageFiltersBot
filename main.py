import discord
from discord.ext import commands, tasks
from discord.ext.commands import cooldown, BucketType
from PIL import Image, ImageDraw
import itertools
import numpy
import pygame
import pygame.gfxdraw
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
from collections import defaultdict
import io
import os
import functools
import typing
import asyncio
import psutil
import math
import redis

#tokens

redis_server = redis.Redis()
AUTH_TOKEN = str(redis_server.get('AUTH_TOKEN').decode('utf-8'))

#setup
#Variables

queue = []
queue_open = None
grayscale_array = [0.2126, 0.7152, 0.0722] #red, green, and blue
status = "t trianglify"
warning_loopt = itertools.cycle(["!", "¡"])

#pip.main(["install", "Pillow", "--upgrade"])

intents = discord.Intents(messages = True, members = True, guilds = True)
bot = commands.Bot(command_prefix = ["t ", "T "], intents = intents, case_insensitive = True, help_command = None)

#functions

def to_thread(func: typing.Callable) -> typing.Coroutine:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        wrapped = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, func, *args, **kwargs)
    return wrapper

def check_for_open_queue():
    if queue_open == True:
        return True
    else: return False

def toFile(image):
    with io.BytesIO() as image_binary:
        image.save(image_binary, 'PNG')
        image_binary.seek(0)
        return discord.File(fp=image_binary, filename='image.png')

def antialias(imagepath):
    image = Image.open(imagepath)
    w, h = image.size
    image.resize((w, h), resample = Image.ANTIALIAS)
    return image

@to_thread
def trianglify_main(image, userid):
    imgco = pygame.image.load(image)
    inps = pygame.transform.smoothscale(imgco, (imgco.get_width() / 2, imgco.get_height() / 2))
    inp = pygame.surfarray.pixels3d(inps)
    pcw = numpy.array(grayscale_array)
    gs = (inp * pcw).sum(axis =- 1)

    x = gaussian_filter(gs, 2, mode = "reflect")
    x2 = gaussian_filter(gs, 30, mode = "reflect")

    diff = (x - x2)
    diff[diff < 0] *= 0.1
    diff = numpy.sqrt(numpy.abs(diff) / diff.max())

    def sample(ref, n = 1000000): #big number 1 million
        numpy.random.seed(0)
        w, h = x.shape
        xs = numpy.random.randint(0, w, size = n)
        ys = numpy.random.randint(0, h, size = n)
        value = ref[xs, ys]
        accept = numpy.random.random(size = n) < value
        points = numpy.array([xs[accept], ys[accept]])
        return points.T, value[accept]

    samples, v = sample(diff)
    plt.scatter(samples[:, 0], -samples[:, 1], c = v, s = 0.2, edgecolors = "none", cmap = "viridis")

    def get_c(tri, image):
        colours = defaultdict(lambda: [])
        w, h, _ = image.shape
        for i in range(0, w):
            for j in range(0, h):
                index = tri.find_simplex((i, j))
                colours[int(index)].append(inp[i, j, :])
        for index, array in colours.items():
            colours[index] = numpy.array(array).mean(axis = 0)
        return colours

    def draw(tri, colours, screen, upscale):
        s = screen.copy()
        for key, c in colours.items():
            t = tri.points[tri.simplices[key]]
            pygame.gfxdraw.filled_polygon(s, t * upscale, c)
            pygame.gfxdraw.polygon(s, t * upscale, c)
        return s

    w, h, _ = inp.shape
    upscale = 2
    screen = pygame.Surface((w * upscale, h * upscale))
    screen.fill(inp.mean(axis = (0, 1)))
    corners = numpy.array([(0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1)])
    points = numpy.concatenate((corners, samples))

    for i in range(0, 25):
        n = 5 + i + 2 * int(i ** 2)
        tri = Delaunay(points[:n, :])
        colours = get_c(tri, inp)
        s = draw(tri, colours, screen, upscale)
        s = pygame.transform.smoothscale(s, (w * 2, h * 2))
        yield i
    pygame.image.save(s, "./finishedImages/imageTri_{}.png".format(userid))
    yield "a string for yield"

#events

@bot.event
async def on_ready():
    change_status.start()
    print("TrianglificatorBot has started.")

@tasks.loop(seconds=10)
async def change_status():
    await bot.change_presence(activity = discord.Game(status))

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandOnCooldown):
        u = ctx.author
        m = "You are currently on cooldown. Please wait {} seconds before trying again.".format(round(error.retry_after))
        await u.send(m)
    else:
        raise(error)

#commands

@bot.command(name = "trianglify")
@commands.cooldown(1, 10, commands.BucketType.user)
async def trianglify(ctx):
    channel = ctx.channel
    await ctx.send("Send the file you would like to trianglify. Supported image types: PNG, JPG, JPEG, BMP, SVG")
    # check if channel is correct and author is the same
    def check(au):
        def i_check(m):
            return m.channel == channel and m.author == au
        return i_check
    try:
        msg = await bot.wait_for("message", check = check(ctx.author), timeout = 60)
        # check if message contains attachment
        if msg.attachments:
            # check if attachment is an image: png, jpg, jpeg, bmp, or svg
            if msg.attachments[0].filename.lower().endswith("png") or msg.attachments[0].filename.lower().endswith("jpg") or msg.attachments[0].filename.lower().endswith("jpeg") or msg.attachments[0].filename.lower().endswith("bmp") or msg.attachments[0].filename.lower().endswith("svg"):
                if not os.path.exists("./userImages/image_{}.png".format(ctx.author.id)):
                    # save the image for the file path
                    await msg.attachments[0].save("./userImages/image_{}.png".format(ctx.author.id))
                    if os.stat("./userImages/image_{}.png".format(ctx.author.id)).st_size <= 2000000:
                        if str(ctx.author.id) not in queue:
                            #pos = queue.index(str(user.id)) + 1
                            uid = int(queue[0])
                            if check_for_open_queue():
                                queue.append(str(ctx.author.id))
                                
                                posfq = queue.index(str(ctx.author.id))
                                if (posfq + 1) > 1:
                                    await ctx.reply("You have been queued and your place is {}.".format(queue[str(ctx.author.id)]))

                                if len(queue) == 1:
                                    while len(queue) >= 1:
                                        
                                        #u = bot.get_user(uid)
                                        pos = queue.index(str(uid)) + 1

                                        if pos == 1:
                                            imgpath = "./userImages/image_{}.png".format(uid)
                                            msg1 = await ctx.send("Currently trianglifying image. The time it takes to render will depend on the image resolution, complexity, and size. WARNING: Using a small image resolution will result in grainy and bad results. <@{}>".format(uid))
                                            func = await trianglify_main(imgpath, uid)
                                            for i in func:
                                                yielded = i
                                                if isinstance(yielded, int):
                                                    await msg1.edit(content = "Currently trianglifying image. The time it takes to render will depend on the image resolution, complexity, and size. {} WARNING {}: Using a small image resolution will result in grainy and bad results. {}% finished. <@{}>".format(next(warning_loopt), next(warning_loopt), (yielded+1)*4, uid))
                                                if isinstance(yielded, str):
                                                    await ctx.reply(file = discord.File("./finishedImages/imageTri_{}.png".format(uid)))
                                            # remove files
                                            if os.path.exists("./userImages/image_{}.png".format(uid)):
                                                os.remove("./userImages/image_{}.png".format(uid))
                                            if os.path.exists("./finishedImages/imageTri_{}.png".format(uid)):
                                                os.remove("./finishedImages/imageTri_{}.png".format(uid))
                                            queue.remove(str(uid))
                                        else:
                                            if os.path.exists("./userImages/image_{}.png".format(uid)):
                                                os.remove("./userImages/image_{}.png".format(uid))
                            else:
                                if os.path.exists("./userImages/image_{}.png".format(uid)):
                                    os.remove("./userImages/image_{}.png".format(uid))
                                await ctx.reply("Queue is currently closed.")

                            
                        else:
                            await ctx.reply("You are already in the queue.")
                    else:
                        if os.path.exists("./userImages/image_{}.png".format(ctx.author.id)):
                            os.remove("./userImages/image_{}.png".format(ctx.author.id))
                        await ctx.reply("The file is too large to trianglify! Max size is 2 MB.")
                else:
                    await ctx.reply("You currently have an image being trianglified. To keep the bot from breaking, you can only have one image being trianglified at once.")
            # error checking
            else:
                await ctx.reply("Please send a PNG, JPG, JPEG, BMP, or SVG file.")
        if not msg.attachments:
            await ctx.reply("Please send a file to trianglify.")
    except asyncio.TimeoutError:
        await ctx.reply("You did not respond in time.")

@bot.command(name = "rusage")
@commands.cooldown(1, 10, commands.BucketType.user)
async def resourceUsage(ctx):
    r = "CPU percent usage: {}%\nCPU speed: about {} GHz\nRAM percent usage: {}%\nTotal RAM used: about {} GB\nTotal RAM: about {} GB".format(psutil.cpu_percent(4), round(psutil.cpu_freq().current/1000, 1), psutil.virtual_memory()[2], round(psutil.virtual_memory()[3]/1000000000, 2), round(psutil.virtual_memory()[0]/1000000000))
    await ctx.reply(r)

@bot.command(name = "queue")
@commands.cooldown(1, 10, commands.BucketType.user)
async def queuePos(ctx):
    if str(ctx.author.id) in queue:
        pos = queue.index(str(ctx.author.id)) + 1
        try:
            if pos == 1:
                await ctx.reply("You are 1st in the queue. Your image is currently being trianglified.")
            else:
                await ctx.reply("Your place in the queue is {}.".format(pos))
        except ValueError:
            await ctx.reply("An error has occurred.")
    else:
        await ctx.reply("You are not in the queue!")

@bot.command(name = "queue_open")
@commands.cooldown(1, 10, commands.BucketType.user)
async def queueAvailability(ctx, args):
    if ctx.author.id == (488730568209465344):
        if args == "true":
            queue_open = True
            print(queue_open)
            await ctx.reply("Successfully changed.")
        if args == "false":
            queue_open = False
            print(queue_open)
            await ctx.reply("Successfully changed.")

bot.run(AUTH_TOKEN)