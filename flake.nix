{
  description = "tiny-llm-gate — a memory-conscious OpenAI-compatible LLM gateway";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        version =
          if self ? shortRev then "0.3.0-${self.shortRev}"
          else if self ? dirtyShortRev then "0.3.0-${self.dirtyShortRev}"
          else "0.3.0-dev";
      in
      {
        packages.default = pkgs.buildGoModule {
          pname = "tiny-llm-gate";
          inherit version;
          src = ./.;
          vendorHash = "sha256-g+yaVIx4jxpAQ/+WrGKxhVeliYx7nLQe/zsGpxV4Fn4=";

          ldflags = [
            "-s"
            "-w"
            "-X main.Version=${version}"
          ];

          # Keep the binary small and statically linked.
          env.CGO_ENABLED = "0";

          meta = with pkgs.lib; {
            description = "Memory-conscious OpenAI-compatible LLM gateway";
            homepage = "https://github.com/nSimonFR/tiny-llm-gate";
            license = licenses.mit;
            mainProgram = "tiny-llm-gate";
          };
        };

        devShells.default = pkgs.mkShell {
          packages = [ pkgs.go pkgs.gopls pkgs.gotools ];
        };

        apps.default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/tiny-llm-gate";
        };
      }) // {
      nixosModules.default = import ./nixos-module.nix;
      nixosModules.tiny-llm-gate = import ./nixos-module.nix;
    };
}
