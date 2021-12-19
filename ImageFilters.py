# install extensions

import discord
from discord.ext import commands, tasks
from discord.ext.commands import cooldown, BucketType

from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
import pygame
import pygame.gfxdraw

import itertools
import numpy as np
import functools
import math

from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter

from collections import defaultdict

import io
import os
import typing
import asyncio
import psutil

import mainpoly

import redis

#tokens

redis_server = redis.Redis()
AUTH_TOKEN = str(redis_server.get('AUTH_TOKEN').decode('utf-8'))

#setup
#Variables

queue = []
queue_open = open("./queue_open/value.txt", "r")
grayscale_array = [0.2126, 0.7152, 0.0722] #red, green, and blue
status = "use: i help"
warning_loopt = itertools.cycle(["!", "¡"])

# bot setup

intents = discord.Intents(messages = True, members = True, guilds = True)
bot = commands.Bot(command_prefix = ["i ", "I "], intents = intents, case_insensitive = True, help_command = None)

#functions

# this function makes other functions run on different threads if using
# @to_thread before function
def to_thread(func: typing.Callable) -> typing.Coroutine:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        wrapped = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, func, *args, **kwargs)
    return wrapper

# check for file and remove

def check_remove(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)


#checking to see if queue is open

def check_for_open_queue():
    f = open("./queue_open/value.txt", "r")
    c = f.read()
    if c == "True":
        return True
    if c == "False":
        return False

# turning an image into a file type
# note to self: BytesIO runs off memory, maybe try to find a better method

def toFile(image):
    with io.BytesIO() as image_binary:
        image.save(image_binary, 'PNG')
        image_binary.seek(0)
        return discord.File(fp=image_binary, filename='image.png')

# image antialiasing

def antialias(imagepath):
    image = Image.open(imagepath)
    w, h = image.size
    image.resize((w, h), resample = Image.ANTIALIAS)
    return image

# main function to trianglify images

@to_thread
def trianglify_main(image, userid):
    imgco = pygame.image.load(image)
    inps = pygame.transform.smoothscale(imgco, (imgco.get_width() / 2, imgco.get_height() / 2))
    inp = pygame.surfarray.pixels3d(inps)
    pcw = np.array(grayscale_array)
    gs = (inp * pcw).sum(axis =- 1)

    x = gaussian_filter(gs, 2, mode = "reflect")
    x2 = gaussian_filter(gs, 30, mode = "reflect")

    diff = (x - x2)
    diff[diff < 0] *= 0.1
    diff = np.sqrt(np.abs(diff) / diff.max())

    def sample(ref, n = 1000000): #big number 1 million
        np.random.seed(0)
        w, h = x.shape
        xs = np.random.randint(0, w, size = n)
        ys = np.random.randint(0, h, size = n)
        value = ref[xs, ys]
        accept = np.random.random(size = n) < value
        points = np.array([xs[accept], ys[accept]])
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
            colours[index] = np.array(array).mean(axis = 0)
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
    corners = np.array([(0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1)])
    points = np.concatenate((corners, samples))

    for i in range(0, 25):
        n = 5 + i + 2 * int(i ** 2)
        tri = Delaunay(points[:n, :])
        colours = get_c(tri, inp)
        s = draw(tri, colours, screen, upscale)
        s = pygame.transform.smoothscale(s, (w * 2, h * 2))
        yield i
    pygame.image.save(s, "./finishedImages/imageTri_{}.png".format(userid))
    yield "a string for yield"


# blur functions

def simple_blur(ipath, uid):
    img = Image.open(ipath)
    blurred = img.filter(ImageFilter.BLUR)
    blurred.save("./finishedImages/imageBlurFinished_{}.png".format(uid))

def box_blur(ipath, uid, radius):
    img = Image.open(ipath)
    boxblurred = img.filter(ImageFilter.BoxBlur(radius))
    boxblurred.save("./finishedImages/imageBlurFinished_{}.png".format(uid))

def gaussian_blur(ipath, uid, radius):
    img = Image.open(ipath)
    boxblurred = img.filter(ImageFilter.GaussianBlur(radius))
    boxblurred.save("./finishedImages/imageBlurFinished_{}.png".format(uid))

#events

@bot.event
async def on_ready():
    await bot.change_presence(activity = discord.Game(status))
    print("{} has started.".format(bot.user.name))

    for v in os.listdir("./userImages"):
        os.remove("./userImages/{}".format(v))
    for v in os.listdir("./finishedImages"):
        os.remove("./finishedImages/{}".format(v))


#change status

#checking for command errors

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandOnCooldown):
        u = ctx.author
        m = "You are currently on cooldown. Please wait {} seconds before trying again.".format(round(error.retry_after))
        await u.send(m)
    else:
        raise(error)

#commands

@bot.command(name = "help")
@commands.cooldown(1, 5, commands.BucketType.user)
async def helpc(ctx):
    clist = ["trianglify", "queue", "grayscale", "pixelate"]
    cdesc = ["Turn an image into a triangle pattern of it.", "See what place in the queue you are for trianglification.", "Turn images into black and white.", "Pixelate images."]

    embed = discord.Embed(title = "Help", description = "-------------")
    
    for v in clist:
        index = clist.index(v)
        embed.add_field(name = "**{}**".format(v), value = "*{}*".format(cdesc[index]), inline = False)

    await ctx.reply(embed = embed)
    

@bot.command(name = "trianglify")
@commands.cooldown(1, 10, commands.BucketType.user)
async def trianglify(ctx):
    channel = ctx.channel
    await ctx.reply("Send the file you would like to trianglify. Supported image types: PNG, JPG, JPEG, BMP, SVG")
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
                if not os.path.exists("./userImages/imageT_{}.png".format(ctx.author.id)):
                    # save the image for the file path
                    await msg.attachments[0].save("./userImages/imageT_{}.png".format(ctx.author.id))
                    if os.stat("./userImages/imageT_{}.png".format(ctx.author.id)).st_size <= 2000000:
                        if str(ctx.author.id) not in queue:
                            #pos = queue.index(str(user.id)) + 1
                            
                            if check_for_open_queue():
                                queue.append(str(ctx.author.id))
                                
                                posfq = queue.index(str(ctx.author.id))
                                if (posfq + 1) > 1:
                                    await ctx.reply("You have been queued and your place is {}.".format(queue[str(ctx.author.id)]))

                                if len(queue) == 1:
                                    while len(queue) >= 1:
                                        uid = int(queue[0])
                                        #u = bot.get_user(uid)
                                        pos = queue.index(str(uid)) + 1

                                        if pos == 1:
                                            try:
                                                imgpath = "./userImages/imageT_{}.png".format(uid)
                                                msg1 = await ctx.send("Currently trianglifying image. The time it takes to render will depend on the image resolution, complexity, and size. WARNING: Using a small image resolution will result in grainy and bad results. <@{}>".format(uid))
                                                func = await trianglify_main(imgpath, uid)
                                                for i in func:
                                                    yielded = i
                                                    if isinstance(yielded, int):
                                                        await msg1.edit(content = "Currently trianglifying image. The time it takes to render will depend on the image resolution, complexity, and size. {} WARNING {}: Using a small image resolution will result in grainy and bad results. {}% finished. <@{}>".format(next(warning_loopt), next(warning_loopt), (yielded+1)*4, uid))
                                                    if isinstance(yielded, str):
                                                        await ctx.reply(file = discord.File("./finishedImages/imageTri_{}.png".format(uid)))
                                                # remove files
                                                check_remove("./userImages/imageT_{}.png".format(ctx.author.id))
                                                check_remove("./finishedImages/imageTri_{}.png".format(ctx.author.id))
                                                queue.remove(str(uid))
                                            except ValueError:
                                                await msg1.edit("Please send an image with 24 bit depth or 32 bit depth.")
                                                check_remove("./userImages/imageT_{}.png".format(ctx.author.id))
                                                queue.remove(str(uid))
                                                break
                                        else:
                                            check_remove("./userImages/imageT_{}.png".format(ctx.author.id))
                            else:
                                check_remove("./userImages/imageT_{}.png".format(ctx.author.id))
                                await ctx.reply("Queue is currently closed.")

                            
                        else:
                            check_remove("./userImages/imageT_{}.png".format(ctx.author.id))
                            await ctx.reply("You are already in the queue.")

                    else:
                        check_remove("./userImages/imageT_{}.png".format(ctx.author.id))
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
        check_remove("./userImages/imageT_{}.png".format(ctx.author.id))
        check_remove("./finishedImages/imageTri_{}.png".format(ctx.author.id))

@bot.command(name = "grayscale")
@commands.cooldown(1, 10, commands.BucketType.user)
async def grayscale(ctx):
    channel = ctx.channel
    await ctx.reply("Send the file you would like to be grayscaled. Supported image types: PNG, JPG, JPEG, BMP, SVG.")
    def check(au):
        def i_check(m):
            return m.channel == channel and m.author == au
        return i_check
    try:
        msg = await bot.wait_for("message", check = check(ctx.author), timeout = 60)
        if msg.attachments:
            if msg.attachments[0].filename.lower().endswith("png") or msg.attachments[0].filename.lower().endswith("jpg") or msg.attachments[0].filename.lower().endswith("jpeg") or msg.attachments[0].filename.lower().endswith("bmp") or msg.attachments[0].filename.lower().endswith("svg"):
                if not os.path.exists("./userImages/imageGray_{}.png".format(ctx.author.id)):
                    await msg.attachments[0].save("./userImages/imageGray_{}.png".format(ctx.author.id))
                    if os.stat("./userImages/imageGray_{}.png".format(ctx.author.id)).st_size <= 10000000:
                        i = Image.open("./userImages/imageGray_{}.png".format(ctx.author.id))
                        ig = i.convert("L")
                        ig.save("./finishedImages/imageGrayFinished_{}.png".format(ctx.author.id))
                        
                        await ctx.reply(file = discord.File("./finishedImages/imageGrayFinished_{}.png".format(ctx.author.id)))

                        check_remove("./userImages/imageGray_{}.png".format(ctx.author.id))
                        check_remove("./finishedImages/imageGrayFinished_{}.png".format(ctx.author.id))

                    else:
                        check_remove("./userImages/imageGray_{}.png".format(ctx.author.id))
                        await ctx.reply("File size is too large! Max size is 10 MB.")
                else:
                    await ctx.reply("You currently have an image being grayscaled. Try again later.")
            else:
                await ctx.reply("Please send a PNG, JPG, JPEG, BMP, or SVG file.")
        if not msg.attachments:
            await ctx.reply("Please send a file to grayscale.")
    except asyncio.TimeoutError:
        await ctx.reply("You did not respond in time.")
        check_remove("./userImages/imageGray_{}.png".format(ctx.author.id))
        check_remove("./finishedImages/imageGrayFinished_{}.png".format(ctx.author.id))

@bot.command(name = "pixelate")
@commands.cooldown(1, 10, commands.BucketType.user)
async def pixelate(ctx):
    channel = ctx.channel
    await ctx.reply("Send the file you would like to be pixelate. Supported image types: PNG, JPG, JPEG, BMP, SVG.")
    def check(au):
        def i_check(m):
            return m.channel == channel and m.author == au
        return i_check
    try:
        msg = await bot.wait_for("message", check = check(ctx.author), timeout = 60)
        if msg.attachments:
            if msg.attachments[0].filename.lower().endswith("png") or msg.attachments[0].filename.lower().endswith("jpg") or msg.attachments[0].filename.lower().endswith("jpeg") or msg.attachments[0].filename.lower().endswith("bmp") or msg.attachments[0].filename.lower().endswith("svg"):
                if not os.path.exists("./userImages/imagePix_{}.png".format(ctx.author.id)):
                    await msg.attachments[0].save("./userImages/imagePix_{}.png".format(ctx.author.id))
                    if os.stat("./userImages/imagePix_{}.png".format(ctx.author.id)).st_size <= 10000000:
                        def check2(au):
                            def i_check2(m):
                                return m.channel == channel and m.author == au
                            return i_check2
                        await ctx.reply("Please send the width and height you would like: <width>, <height>. Example: 32, 32 would mean that the image would have 32 pixels as the width and height.")
                        msg2 = await bot.wait_for("message", check = check2(ctx.author), timeout = 30)
                        try:

                            try:
                                uw, uh = int(msg2.content.split(", ")[0]), int(msg2.content.split(", ")[1])
                            except ValueError:
                                await ctx.reply("Please send the dimensions in the format of width, height.")

                            i = Image.open("./userImages/imagePix_{}.png".format(ctx.author.id))
                            iw, ih = i.size
                            if (int(uw) < iw) and (int(uh) < ih):
                                ims = i.resize((int(uw), int(uh)), resample = Image.BILINEAR)
                                imrs = ims.resize(i.size, Image.NEAREST)
                                imrs.save("./finishedImages/imagePixFinished_{}.png".format(ctx.author.id))
                                
                                await ctx.reply(file = discord.File("./finishedImages/imagePixFinished_{}.png".format(ctx.author.id)))
                            else:
                                if (uw >= iw) or (uh >= ih):
                                    await ctx.reply("You can only pixelate an image to less than its original size. The image size is {}, {} and you wanted to resize it to {}, {}.".format(i.size[0], i.size[1], uw, uh))

                            check_remove("./userImages/imagePix_{}.png".format(ctx.author.id))
                            check_remove("./finishedImages/imagePixFinished_{}.png".format(ctx.author.id))

                        except asyncio.TimeoutError:
                            check_remove("./userImages/imagePix_{}.png".format(ctx.author.id))
                            check_remove("./finishedImages/imagePixFinished_{}.png".format(ctx.author.id))
                            await ctx.reply("You did not respond in time.")
                    else:
                        check_remove("./userImages/imagePix_{}.png".format(ctx.author.id))
                        await ctx.reply("File size is too large! Max size is 10 MB.")
                else:
                    await ctx.reply("You currently have an image being pixelated. Try again later.")
            else:
                await ctx.reply("Please send a PNG, JPG, JPEG, BMP, or SVG file.")
        if not msg.attachments:
            await ctx.reply("Please send a file to pixelate.")
    except asyncio.TimeoutError:
        check_remove("./userImages/imagePix_{}.png".format(ctx.author.id))
        check_remove("./finishedImages/imagePixFinished_{}.png".format(ctx.author.id))
        await ctx.reply("You did not respond in time.")

@bot.command(name = "blur")
@commands.cooldown(1, 10, commands.BucketType.user)
async def blur(ctx):
    channel = ctx.channel
    await ctx.reply("Send the file you would like to be blurred. Supported image types: PNG, JPG, JPEG, BMP, SVG.")
    def check(au):
        def i_check(m):
            return m.channel == channel and m.author == au
        return i_check
    try:
        msg = await bot.wait_for("message", check = check(ctx.author), timeout = 60)
        if msg.attachments:
            if msg.attachments[0].filename.lower().endswith("png") or msg.attachments[0].filename.lower().endswith("jpg") or msg.attachments[0].filename.lower().endswith("jpeg") or msg.attachments[0].filename.lower().endswith("bmp") or msg.attachments[0].filename.lower().endswith("svg"):
                if not os.path.exists("./userImages/imageBlur_{}.png".format(ctx.author.id)):
                    await msg.attachments[0].save("./userImages/imageBlur_{}.png".format(ctx.author.id))
                    if os.stat("./userImages/imageBlur_{}.png".format(ctx.author.id)).st_size <= 10000000:
                        
                        try:
                            await ctx.reply("What type of blur do you want? s for simple blur, g for gaussian blur, and b for box blur.")
                            def check2(au):
                                def i_check2(m):
                                    return m.channel == channel and m.author == au
                                return i_check2

                            msg2 = await bot.wait_for("message", check = check2(ctx.author), timeout = 15)
                            
                            if msg2.content == "s" or msg2.content == "S":
                                simple_blur("./userImages/imageBlur_{}.png".format(ctx.author.id), ctx.author.id)
                            if msg2.content == "b" or msg2.content == "B":
                                try:
                                    def check3(au):
                                        def i_check3(m):
                                            return m.channel == channel and m.author == au
                                        return i_check3
                                
                                    await ctx.reply("How intense do you want the blur to be? 0 = no blur, and the higher the number, the more the blur.")
                                    msg3 = await bot.wait_for("message", check = check3(ctx.author), timeout = 15)

                                    if isinstance(int(msg3.content), int):
                                        if int(msg3.content) > 0:
                                            box_blur("./userImages/imageBlur_{}.png".format(ctx.author.id), ctx.author.id, int(msg3.content))
                                        else:
                                            await ctx.reply("Please send a number larger than 0.")
                                            check_remove("./userImages/imageBlur_{}.png".format(ctx.author.id))
                                            return
                                except ValueError:
                                    await ctx.reply("Please send a number.")
                            if msg2.content == "g" or msg2.content == "G":
                                try:
                                    def check3_2(au):
                                        def i_check3_2(m):
                                            return m.channel == channel and m.author == au
                                        return i_check3_2
                                
                                    await ctx.reply("How intense do you want the blur to be? 0 = no blur, and the higher the number, the more the blur.")
                                    msg3_2 = await bot.wait_for("message", check = check3_2(ctx.author), timeout = 15)

                                    if isinstance(int(msg3_2.content), int):
                                        if int(msg3_2.content) > 0:
                                            gaussian_blur("./userImages/imageBlur_{}.png".format(ctx.author.id), ctx.author.id, int(msg3_2.content))
                                        else:
                                            await ctx.reply("Please send a number larger than 0.")
                                            check_remove("./userImages/imageBlur_{}.png".format(ctx.author.id))
                                            return
                                except ValueError:
                                    await ctx.reply("Please send a number.")
                            
                            if os.path.exists("./finishedImages/imageBlurFinished_{}.png".format(ctx.author.id)):
                                await ctx.reply(file = discord.File("./finishedImages/imageBlurFinished_{}.png".format(ctx.author.id)))
                            else:
                                await ctx.reply("Please send either s, g, or b.")

                            check_remove("./userImages/imageBlur_{}.png".format(ctx.author.id))
                            check_remove("./finishedImages/imageBlurFinished_{}.png".format(ctx.author.id))
                        except ValueError:
                            check_remove("./userImages/imageBlur_{}.png".format(ctx.author.id))
                            await ctx.reply("An error occurred.")
                    else:
                        check_remove("./userImages/imageBlur_{}.png".format(ctx.author.id))
                        await ctx.reply("File size is too large! Max size is 10 MB.")
                else:
                    await ctx.reply("You currently have an image being blurred. Try again later.")
            else:
                await ctx.reply("Please send a PNG, JPG, JPEG, BMP, or SVG file.")
        if not msg.attachments:
            await ctx.reply("Please send a file to blur.")
    except asyncio.TimeoutError:
        await ctx.reply("You did not respond in time.")
        check_remove("./userImages/imageBlur_{}.png".format(ctx.author.id))
        check_remove("./finishedImages/imageBlurFinished_{}.png".format(ctx.author.id))


@bot.command(name = "runwp")
async def runwp(ctx):
    rgbval = [list(np.random.choice(range(255), size=3)), list(np.random.choice(range(255), size=3))]
    mainpoly.mainfunc(rgbval, np.random.randint(5, 60), [1920, 1080], (200, 200))
    if os.path.exists("./renders/wallpaper.jpg"):
        await ctx.send(file = discord.File("./renders/wallpaper.jpg"))
        check_remove("./renders/wallpaper.jpg")

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

@bot.command(name = "shutdown")
async def shutdown(ctx):
    if ctx.author.id == (488730568209465344):
        await ctx.reply("Shutting down...")
        await bot.close()

@bot.command(name = "queue_open")
@commands.cooldown(1, 10, commands.BucketType.user)
async def queueAvailability(ctx, args):
    if ctx.author.id == (488730568209465344):
        if args == "true":
            f = open("./queue_open/value.txt", "w")
            f.write("True")
            await ctx.reply("Successfully changed.")
        if args == "false":
            f = open("./queue_open/value.txt", "w")
            f.write("False")
            await ctx.reply("Successfully changed.")

bot.run(AUTH_TOKEN)