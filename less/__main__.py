from click import group, argument


@group()
def main():
    pass


@main.command()
@argument('text', type = str)
def act(text: str):
    print(text)


if __name__ == '__main__':
    main()
