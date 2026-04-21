{ config, lib, pkgs, ... }:

let
  cfg = config.services.tiny-llm-gate;
  configFormat = pkgs.formats.yaml { };
  configFile =
    if cfg.configFile != null then cfg.configFile
    else configFormat.generate "tiny-llm-gate-config.yaml" cfg.settings;
in
{
  options.services.tiny-llm-gate = {
    enable = lib.mkEnableOption "tiny-llm-gate (OpenAI-compatible LLM gateway)";

    package = lib.mkOption {
      type = lib.types.package;
      description = "tiny-llm-gate package to use.";
    };

    settings = lib.mkOption {
      type = configFormat.type;
      default = { };
      description = ''
        YAML config as a Nix attrset. Ignored if `configFile` is set.
        See `testdata/example-config.yaml` in the repo for the schema.
      '';
    };

    configFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Path to a YAML config file. If null, generated from `settings`.";
    };

    memoryMax = lib.mkOption {
      type = lib.types.str;
      default = "30M";
      description = "systemd MemoryMax hard ceiling.";
    };

    goMemLimit = lib.mkOption {
      type = lib.types.str;
      default = "20MiB";
      description = "GOMEMLIMIT env — the Go runtime's soft memory target.";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.services.tiny-llm-gate = {
      description = "tiny-llm-gate — OpenAI-compatible LLM gateway";
      after = [ "network.target" ];
      wantedBy = [ "multi-user.target" ];

      environment = {
        GOMEMLIMIT = cfg.goMemLimit;
        # Aggressive GC to minimize RSS.
        GOGC = "50";
      };

      serviceConfig = {
        ExecStart = "${cfg.package}/bin/tiny-llm-gate --config ${configFile}";
        Restart = "on-failure";
        RestartSec = "5s";

        # Hard memory ceiling — OOMs if we regress.
        MemoryMax = cfg.memoryMax;

        # Sandboxing.
        DynamicUser = true;
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
        PrivateDevices = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictSUIDSGID = true;
        LockPersonality = true;
        RestrictRealtime = true;
        SystemCallArchitectures = "native";
        CapabilityBoundingSet = "";
      };
    };
  };
}
