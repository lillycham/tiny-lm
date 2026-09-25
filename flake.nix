{
  description = "A small character-level language model, from NumPy to PyTorch";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs:
        let
          pythonEnv = pkgs.python3.withPackages (ps: [ ps.numpy ps.torch ]);
        in
        {
          default = pkgs.mkShell {
            packages = [ pythonEnv ];
            # A fixed path to this Python for editors, because the store path changes on every update.
            shellHook = ''
              ln -sfn ${pythonEnv} .python-env
            '';
          };
        });
    };
}
