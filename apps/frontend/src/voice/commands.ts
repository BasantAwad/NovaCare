type CommandAction = (params?: any) => void;

interface Command {
  id: string;
  action: CommandAction;
  description: string;
}

class CommandRegistry {
  private commands: Map<string, Command> = new Map();

  register(id: string, action: CommandAction, description: string) {
    this.commands.set(id, { id, action, description });
  }

  unregister(id: string) {
    this.commands.delete(id);
  }

  execute(id: string, params?: any) {
    const command = this.commands.get(id);
    if (command) {
      command.action(params);
      return true;
    }
    return false;
  }
  
  getAllCommands() {
    return Array.from(this.commands.values());
  }
}

export const commandRegistry = new CommandRegistry();
